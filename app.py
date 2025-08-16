import os
import ast
import numpy as np
import pandas as pd
import streamlit as st
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# 🔐 OpenAI 키: 환경변수 우선 → 없으면 Streamlit secrets
API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not API_KEY:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. Streamlit Secrets(⋮ → Settings → Secrets) 또는 환경변수로 추가하세요.")
    st.stop()

client = OpenAI(api_key=API_KEY)

# 📥 CSV 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("nurse_1_with_embeddings.csv")
    # Embedding 컬럼을 문자열 → 리스트로
    df["Embedding"] = df["Embedding"].apply(ast.literal_eval)
    return df

def embed_text(text: str):
    # 최신 임베딩 엔드포인트 사용
    resp = client.embeddings.create(
        model="text-embedding-3-small",  # 또는 "text-embedding-3-large"
        input=[text]
    )
    return resp.data[0].embedding

def find_most_similar(user_embedding, df):
    all_embeddings = np.array(df["Embedding"].to_list())
    sims = cosine_similarity([user_embedding], all_embeddings)[0]
    best_idx = int(np.argmax(sims))
    return df.iloc[best_idx], float(sims[best_idx])

# ========== Streamlit 시작 ==========
st.set_page_config(page_title="간호사 상황극 문제은행", page_icon="🩺")
st.title("🩺 간호사 100문 100답 - 카테고리 선택 문제은행")

# === 데이터 및 세션 초기화 ===
if "raw_df" not in st.session_state:
    st.session_state.raw_df = load_data()
if "category_selected" not in st.session_state:
    st.session_state.category_selected = "전체"
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = st.session_state.raw_df.copy()
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0
if "total_count" not in st.session_state:
    st.session_state.total_count = 0
if "solved_ids" not in st.session_state:
    st.session_state.solved_ids = []
if "category_stats" not in st.session_state:
    st.session_state.category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
if "quiz_finished" not in st.session_state:
    st.session_state.quiz_finished = False

# === 카테고리 필터 ===
all_categories = set()
for etc in st.session_state.raw_df["Etc"]:
    for e in str(etc).split(";"):
        e = e.strip()
        if e:
            all_categories.add(e)

category_options = ["전체"] + sorted(list(all_categories))
selected = st.selectbox("📂 푸실 문제 카테고리를 선택하세요:", category_options)

# 카테고리 변경 시 필터링
if selected != st.session_state.category_selected:
    st.session_state.category_selected = selected
    if selected == "전체":
        st.session_state.filtered_df = (
            st.session_state.raw_df.sample(frac=1, random_state=None).reset_index(drop=True)
        )
    else:
        mask = st.session_state.raw_df["Etc"].apply(lambda x: selected in str(x))
        st.session_state.filtered_df = (
            st.session_state.raw_df[mask].sample(frac=1, random_state=None).reset_index(drop=True)
        )
    st.session_state.current_idx = 0
    st.session_state.correct_count = 0
    st.session_state.total_count = 0
    st.session_state.solved_ids = []
    st.session_state.category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    st.session_state.quiz_finished = False

df = st.session_state.filtered_df
idx = st.session_state.current_idx

# 퀴즈 완료 여부
if idx >= len(df):
    st.session_state.quiz_finished = True

# ========== 문제 풀이 ==========
if not st.session_state.quiz_finished:
    row = df.iloc[idx]
    st.markdown(f"**문제 {idx + 1}:** {row['Question']}")
    user_input = st.text_area("🧑‍⚕️ 당신의 간호사 응답은?", key=f"input_{idx}_{selected}")

    if st.button("정답 제출") and user_input.strip():
        with st.spinner("AI가 채점 중입니다..."):
            try:
                user_embedding = embed_text(user_input)
                best_match, similarity = find_most_similar(user_embedding, df)

                st.session_state.total_count += 1
                st.session_state.solved_ids.append(idx)

                is_correct = False
                if similarity >= 0.8:
                    st.session_state.correct_count += 1
                    st.success(f"✅ 정답입니다! 유사도 {similarity:.2f}")
                    is_correct = True
                elif similarity >= 0.6:
                    st.info(f"🟡 거의 맞았어요! 유사도 {similarity:.2f}")
                else:
                    st.error(f"❌ 오답입니다. 유사도 {similarity:.2f}")

                st.markdown(f"**정답 예시:**\n> {best_match['Answer']}")
                st.caption(f"🗂️ 카테고리: {str(best_match['Etc'])}")

                for category in str(best_match["Etc"]).split(";"):
                    category = category.strip()
                    if category:
                        st.session_state.category_stats[category]["total"] += 1
                        if is_correct:
                            st.session_state.category_stats[category]["correct"] += 1
            except Exception as e:
                st.error(f"채점 중 오류가 발생했습니다: {e}")

    if st.button("다음 문제"):
        st.session_state.current_idx += 1

# ========== 퀴즈 완료 ==========
else:
    st.success("🎉 모든 문제를 완료했습니다!")

    st.subheader("📊 최종 결과 요약")
    correct = st.session_state.correct_count
    total = st.session_state.total_count
    st.markdown(f"- 총 문제 수: **{total}**")
    st.markdown(f"- 맞힌 문제 수: **{correct}**")
    if total > 0:
        st.markdown(f"- 정답률: **{(correct/total)*100:.1f}%**")
    else:
        st.markdown("- 정답률: **0.0%**")

    st.markdown("---")
    st.subheader("🧾 카테고리별 정답 통계")

    stats = st.session_state.category_stats
    for cat, stat in stats.items():
        if stat["total"] > 0:
            rate = stat["correct"] / stat["total"] * 100
            st.write(f"- **{cat}**: {stat['correct']} / {stat['total']} 정답 ({rate:.1f}%)")

    st.markdown("---")
    if st.button("🔁 처음부터 다시 시작하기"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

