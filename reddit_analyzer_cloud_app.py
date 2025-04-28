# ----- BEGINNING OF reddit_analyzer_cloud_app.py -----
# Final Cloud Version v6 - Ensuring consistent space indentation
import streamlit as st
import praw
import json
from datetime import datetime, time
import pandas as pd
import time as py_time
import pytz
import os
import google.generativeai as genai

# --- 页面配置 ---
st.set_page_config(page_title="Reddit 分析器 + AI 对话", layout="wide", initial_sidebar_state="expanded")

# --- 添加图标 ---
st.title(":mag: Reddit 分析器 + AI 对话 :robot_face:")
st.caption("抓取 Reddit 讨论，并通过多轮对话与 Gemini AI 进行分析")

# === Gemini 关键词处理函数 ===
def get_english_keywords_for_query(api_key_used, user_query):
    """使用 Gemini 检测语言，翻译中文，并扩展英文关键词"""
    if not api_key_used:
        st.warning("未配置 Gemini Key，无法自动处理中文关键词。将直接使用原始输入。")
        return user_query
    if not user_query.strip():
         st.warning("输入的关键词为空，无法处理。")
         return ""
    try:
        model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25') # 使用免费实验模型
        prompt = f"""
        Analyze the following user query for a Reddit search: "{user_query}"

        1. Detect the primary language of the query.
        2. If the language is Chinese, translate the core meaning into a concise English keyword or phrase. Also generate 2-3 additional relevant English synonyms or related search terms.
        3. If the language is already English, use the original query as the primary keyword and generate 2-3 additional relevant English synonyms or related search terms based on the core meaning.
        4. Return ONLY a comma-separated list of the final English keywords/phrases (the primary one first, followed by related terms). Ensure each term is relevant for a keyword search. Example output for "老年人 消费": "(elderly consumption), (senior spending), (aging population purchase), (geriatric cost)"
        """
        response = model.generate_content(prompt)
        if hasattr(response, 'text') and response.text:
            keywords_str = response.text.strip()
            keywords_list = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
            if keywords_list:
                if len(keywords_list) == 1:
                    term = keywords_list[0].strip('()')
                    reddit_query = f"({term})" if term else ""
                else:
                    processed_terms = []
                    for kw in keywords_list:
                         term = kw.strip('()')
                         if term: processed_terms.append(f"({term})")
                    reddit_query = " OR ".join(processed_terms) if processed_terms else ""
                original_term_in_parentheses = f"({user_query.strip()})"
                if reddit_query and reddit_query.lower() != original_term_in_parentheses.lower():
                     st.info(f"已将输入自动转换为/扩展为英文 OR 查询: {reddit_query}")
                elif reddit_query:
                     st.info("输入为英文或已处理，将使用以下查询。")
                return reddit_query if reddit_query else user_query
            else:
                st.warning("Gemini 未能提取有效的英文关键词...")
                return user_query
        else:
            st.warning(f"Gemini未能处理关键词...")
            return user_query
    except Exception as e:
        st.error(f"调用 Gemini 处理关键词时出错: {e}")
        return user_query

# === 核心抓取函数 ===
def run_scraper(sub_name, query, limit, t_filter, get_comments, use_dates, start_dt, end_dt):
    """执行 Reddit 数据抓取的函数 (云版本 - 不跳过已处理帖子)"""
    all_data = []
    status_area = st.empty()
    try:
        status_area.info("正在初始化 PRAW 并连接 Reddit API...")
        # !! 使用侧边栏读取的全局变量 !!
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT, read_only=True)
        subreddit = reddit.subreddit(sub_name)
        status_area.info(f"连接成功！正在 r/{sub_name} 中搜索 '{query}' (基本时间范围: {t_filter})...")
        if not query or not query.strip('()'):
            status_area.warning("搜索查询为空...")
            return []
        search_results = list(subreddit.search(query=query, sort='new', time_filter=t_filter, limit=limit))
        status_area.info(f"初步找到 {len(search_results)} 个帖子...")
        start_timestamp, end_timestamp = None, None
        if use_dates and start_dt and end_dt:
            try:
                start_datetime_naive = datetime.combine(start_dt, time.min); end_datetime_naive = datetime.combine(end_dt, time.max)
                utc_tz = pytz.utc; start_timestamp = utc_tz.localize(start_datetime_naive).timestamp(); end_timestamp = utc_tz.localize(end_datetime_naive).timestamp()
                if start_timestamp > end_timestamp: st.warning("开始日期晚于结束日期..."); start_timestamp, end_timestamp = None, None
                else: status_area.info(f"将应用日期范围: {start_dt} 至 {end_dt}")
            except Exception as date_e: st.error(f"处理日期输入时出错: {date_e}"); start_timestamp, end_timestamp = None, None
        total_posts_processed_this_run, date_skipped_count = 0, 0
        progress_bar = st.progress(0) if search_results else None
        for i, submission in enumerate(search_results):
            if progress_bar: progress_bar.progress((i + 1) / len(search_results))
            if start_timestamp and end_timestamp:
                if not (start_timestamp <= submission.created_utc <= end_timestamp): date_skipped_count +=1; continue
            status_area.info(f"正在处理帖子 {i+1}/{len(search_results)}: {submission.id} - {submission.title[:30]}...")
            post_info = {'id': submission.id, 'title': submission.title, 'selftext': submission.selftext, 'url': submission.url, 'score': submission.score, 'created_utc': submission.created_utc, 'comments': []}
            if get_comments:
                try:
                    submission.comment_sort = 'new'; submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list():
                        if hasattr(comment, 'body'): post_info['comments'].append({'id': comment.id, 'author': str(comment.author),'body': comment.body, 'score': comment.score,'created_utc': comment.created_utc, 'parent_id': comment.parent_id, 'depth': comment.depth})
                except Exception as e: st.warning(f"处理帖子 {submission.id} 评论时出错: {e}")
            all_data.append(post_info); total_posts_processed_this_run += 1; py_time.sleep(0.1)
        status_area.success(f"抓取完成！本次运行处理了 {total_posts_processed_this_run} 个帖子。跳过了 {date_skipped_count} 个不符合日期范围的帖子。")
        return all_data
    except praw.exceptions.PRAWException as pe: status_area.error(f"Reddit API (PRAW) 错误: {pe}"); return None
    except Exception as e: status_area.error(f"发生意外错误: {e}"); return None

# === Gemini 分析函数 ===
def generate_gemini_response(api_key_used, chat_history):
    """调用 Gemini API 生成聊天回复"""
    if not api_key_used:
        return "错误：未配置 Gemini API Key。"
    if not chat_history:
        return "错误：聊天历史为空。"
    try:
        model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25') # 使用免费实验模型
        response = model.generate_content(chat_history)
        if hasattr(response, 'text'):
             return response.text
        else:
             feedback = getattr(response, 'prompt_feedback', '无详细反馈')
             st.error(f"Gemini API 返回了响应，但无法提取文本。可能是因为内容安全阻止。反馈: {feedback}")
             return f"错误：Gemini API 返回了无效响应或被阻止。反馈: {feedback}"
    except Exception as e:
        st.error(f"调用 Gemini API 时出错: {e}")
        return f"调用 Gemini API 时出错: {e}"

# --- 侧边栏定义 ---
with st.sidebar:
    st.header("参数配置")
    # --- API 凭证读取 ---
    # Reddit
    try:
        REDDIT_CLIENT_ID = st.secrets["REDDIT_CLIENT_ID"]
        REDDIT_CLIENT_SECRET = st.secrets["REDDIT_CLIENT_SECRET"]
        REDDIT_USER_AGENT = st.secrets["REDDIT_USER_AGENT"]
        if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET or not REDDIT_USER_AGENT:
             st.error("错误：Reddit API 凭证未配置..."); st.stop()
    except KeyError: st.error("错误：请配置 Reddit API 凭证 Secrets。"); st.stop()
    except Exception as e: st.error(f"读取 Reddit Secrets 时出错: {e}"); st.stop()
    # Gemini
    GEMINI_API_KEY = None; gemini_configured_successfully = False
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        if not GEMINI_API_KEY: st.warning("提示：Gemini API Key 未配置...")
        else:
            try: genai.configure(api_key=GEMINI_API_KEY); st.success("Gemini API Key 配置成功！"); gemini_configured_successfully = True
            except Exception as gemini_config_e: st.error(f"配置 Gemini API 时出错: {gemini_config_e}"); GEMINI_API_KEY = None
    except KeyError: st.warning("提示：未找到 Gemini API Key Secret...")
    except Exception as e: st.error(f"读取 Gemini Secrets 时出错: {e}")

    # --- 输入参数 ---
    subreddit_name = st.text_input("Subreddit 名称", "askSingapore")
    search_query_input = st.text_input("搜索关键词 (支持中文)", "老年人 消费")
    post_limit = st.number_input("最大帖子数量", min_value=1, max_value=1000, value=20)
    time_filter_options = {"过去一小时": "hour", "过去一天": "day", "过去一周": "week", "过去一月": "month", "过去一年": "year", "所有时间": "all"}
    selected_time_label = st.selectbox("基本时间范围:", options=list(time_filter_options.keys()), index=5)
    time_filter = time_filter_options[selected_time_label]
    st.markdown("---")
    use_specific_dates = st.checkbox("使用精确日期范围")
    start_date_input = st.date_input("开始日期", value=None, disabled=not use_specific_dates)
    end_date_input = st.date_input("结束日期", value=None, disabled=not use_specific_dates)
    st.caption("*日期筛选在获取帖子后进行。*")
    st.markdown("---")
    fetch_comments = st.checkbox("抓取评论 (可能很慢!)", value=True)
    st.markdown("---")
    # --- 触发抓取按钮 ---
    start_scrape_button = st.button("开始抓取 Reddit 数据", type="primary", icon="🔍")

# --- 初始化 Session State ---
if 'scraped_data' not in st.session_state: st.session_state.scraped_data = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'initial_api_input' not in st.session_state: st.session_state.initial_api_input = None
if 'final_query_used' not in st.session_state: st.session_state.final_query_used = None

# --- 主逻辑：处理抓取按钮点击 ---
if start_scrape_button:
    st.session_state.scraped_data = None; st.session_state.chat_history = []; st.session_state.initial_api_input = None; st.session_state.final_query_used = None
    st.info(f"收到抓取任务，正在处理关键词 '{search_query_input}'...")
    final_reddit_query = search_query_input
    if gemini_configured_successfully:
        with st.spinner("正在使用 AI 处理/翻译关键词..."): final_reddit_query = get_english_keywords_for_query(GEMINI_API_KEY, search_query_input)
    st.session_state.final_query_used = final_reddit_query
    if st.session_state.final_query_used: st.info(f"最终用于 Reddit 搜索的查询语句: '{st.session_state.final_query_used}'")
    else: st.warning("关键词处理后为空..."); st.stop()
    with st.spinner(f"正在 r/{subreddit_name} 中搜索并抓取数据..."):
        scraped_data_result = run_scraper(subreddit_name, st.session_state.final_query_used, post_limit, time_filter, fetch_comments, use_specific_dates, start_date_input, end_date_input)
    st.session_state.scraped_data = scraped_data_result
    st.rerun()

# --- 显示抓取结果和启动分析/对话 ---
# (注意这里的 if/elif/else 结构和缩进)
if st.session_state.scraped_data is not None:
    # --- 创建 Tabs ---
    tab_preview, tab_ai_chat = st.tabs(["📊 数据预览与下载", "💬 AI 对话分析"])

    with tab_preview:
        st.subheader("抓取结果预览 (本次运行找到的所有帖子)")
        st.write(f"共找到 {len(st.session_state.scraped_data)} 个符合条件的帖子记录。")
        if st.session_state.final_query_used: st.write(f"(使用的最终搜索查询: `{st.session_state.final_query_used}`) ")
        if not st.session_state.scraped_data: st.warning("本次运行没有找到符合条件的帖子。")
        else:
            # --- 预览表格 ---
            try:
                posts_df_data = [{k: v for k, v in post.items() if k != 'comments'} for post in st.session_state.scraped_data]
                posts_df = pd.DataFrame(posts_df_data)
                if not posts_df.empty and 'created_utc' in posts_df.columns:
                    posts_df['created_datetime_sgt'] = pd.to_datetime(posts_df['created_utc'], unit='s', utc=True).dt.tz_convert('Asia/Singapore')
                    preview_cols = ['title', 'score', 'created_datetime_sgt', 'url', 'id', 'selftext']; display_cols = [col for col in preview_cols if col in posts_df.columns]
                    st.dataframe(posts_df[display_cols])
                else: st.dataframe(posts_df)
            except Exception as e: st.warning(f"无法将结果格式化为表格显示: {e}")
            # --- 下载按钮 ---
            st.subheader("下载本次抓取的数据 (JSON)")
            try:
                json_string = json.dumps(st.session_state.scraped_data, indent=4, ensure_ascii=False)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_query = "".join(c if c.isalnum() else "_" for c in (st.session_state.final_query_used or search_query_input))[:50]
                download_filename = f"reddit_{subreddit_name}_search_{safe_query}_{timestamp_str}.json"
                st.download_button(label="下载 JSON 数据", data=json_string, file_name=download_filename, mime="application/json")
            except Exception as e: st.error(f"生成 JSON 下载文件时出错: {e}")

    with tab_ai_chat:
        st.subheader("与 Gemini AI 对话分析")
        if not gemini_configured_successfully:
            st.error("未成功配置 Gemini API Key，无法进行 AI 对话。")
        else:
            # --- 持久显示首次分析输入 Expander ---
            if st.session_state.initial_api_input:
                with st.expander("查看首次分析发送给 Gemini 的完整输入 (指令+数据)", expanded=False): st.text(st.session_state.initial_api_input)
                st.markdown("---")
            # --- 初始分析 Prompt 输入和触发按钮 ---
            if not st.session_state.chat_history:
                query_for_prompt = st.session_state.final_query_used or search_query_input
                default_initial_prompt = f"你是一个数据分析助手。请根据以下抓取的关于 '{query_for_prompt}' 的 Reddit 帖子和评论内容，总结主要的讨论观点、用户情绪、可能存在的消费趋势以及亚马逊新加坡站的选品思路，并用表格列出每个选品与Reddit内容的关联性，并按照用户需求的迫切性对选品进行排序。"
                st.write("**初步分析指令 (可编辑):**")
                analysis_prompt_input = st.text_area(label="初步分析指令 (可编辑):", value=default_initial_prompt, height=150, key="initial_prompt_input_area", label_visibility="collapsed")
                if st.button("进行初步 AI 分析", icon=":material/auto_awesome:"):
                    with st.spinner("正在准备初步分析请求..."):
                        analysis_prompt = analysis_prompt_input; combined_text = ""; max_chars = 3000000; current_chars = 0; posts_included_count = 0
                        # --- 填充 combined_text 的循环 ---
                        for post in st.session_state.scraped_data:
                            post_content = f"帖子 ID: {post.get('id', '')}\n标题: {post.get('title', '')}\n正文: {post.get('selftext', '')}\n"
                            if current_chars + len(post_content) >= max_chars: st.warning(f"文本长度达到上限 {max_chars}..."); limit_warning_shown = True; break # Ensure flag logic is here if used
                            combined_text += post_content; current_chars += len(post_content); posts_included_count += 1; comment_count = 0; comments_text = ""
                            for comment in post.get('comments', []):
                                comment_content = f"  评论 (ID: {comment.get('id','')}, Score: {comment.get('score', '')}): {comment.get('body', '')}\n"
                                if current_chars + len(comment_content) >= max_chars: # Use >= for consistency
                                    # if not limit_warning_shown: # Add flag if single warning desired
                                    st.warning(f"文本长度达到上限 {max_chars}..."); limit_warning_shown = True
                                    break
                                comments_text += comment_content; current_chars += len(comment_content); comment_count += 1
                                if comment_count >= 10: break
                            combined_text += comments_text + "---\n"
                        # ----------------------------------
                        full_initial_message_for_api = f"{analysis_prompt}\n\n以下是相关数据：\n\n{combined_text}"
                        history_for_api_call = [{"role": "user", "parts": [full_initial_message_for_api]}]
                        st.write(f"最终准备发送给 Gemini 的文本包含 {posts_included_count} 个帖子（部分评论可能因长度限制被截断）。")
                        st.session_state.initial_api_input = full_initial_message_for_api # 存储首次输入
                    with st.spinner("正在调用 Gemini 进行初步分析..."):
                        initial_response_text = generate_gemini_response(GEMINI_API_KEY, history_for_api_call)
                        st.session_state.chat_history = [] # 清空旧历史记录
                        st.session_state.chat_history.append({"role": "user", "parts": [analysis_prompt]}) # 只存指令
                        st.session_state.chat_history.append({"role": "model", "parts": [initial_response_text]}) # 存回复
                        st.rerun() # Rerun 以显示聊天记录和持久的 Expander
            # --- 显示聊天记录 ---
            if st.session_state.chat_history:
                st.markdown("---"); st.write("**AI 对话分析记录:**")
                for message in st.session_state.chat_history:
                     with st.chat_message(message["role"]): st.markdown(message["parts"][0])
            # --- 聊天输入框 ---
            if prompt := st.chat_input("就分析结果或数据进行提问...", disabled=not gemini_configured_successfully):
                 st.session_state.chat_history.append({"role": "user", "parts": [prompt]})
                 with st.chat_message("user"): st.markdown(prompt)
                 with st.chat_message("model"):
                     with st.spinner("Gemini 正在思考..."):
                         response = generate_gemini_response(GEMINI_API_KEY, st.session_state.chat_history)
                         if isinstance(response, str) and not response.startswith("错误："):
                             st.markdown(response); st.session_state.chat_history.append({"role": "model", "parts": [response]})
                         else:
                             if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user": st.session_state.chat_history.pop() # Pop user message if AI fails
                 st.rerun()

elif start_scrape_button: # <--- 检查这里的对齐
    st.error("未能成功抓取数据，请检查侧边栏配置或 API 状态。")
else: # <--- 检查这里的对齐
    st.write("请在左侧配置参数并点击“开始抓取 Reddit 数据”按钮。")

# --- 页脚 ---
st.markdown("---")
st.caption("一个使用 Streamlit 和 PRAW 构建的 Reddit 分析工具 (Cloud + AI Chat + Keyword Processing + Tabs/Icons)")
# ----- END OF reddit_analyzer_cloud_app.py -----