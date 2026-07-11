import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="LoL 챔피언 추천 시스템",
    page_icon="🎮",
    layout="wide"
)

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv("LoL_챔피언.csv")

    numeric_cols = [
        "체력",
        "체력성장",
        "마나",
        "마나성장",
        "이동속도",
        "방어력",
        "마법저항력",
        "공격사거리",
        "공격력",
        "공격력성장",
        "공격속도성장(%)"
    ]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[numeric_cols])

    similarity = cosine_similarity(scaled)

    return df, similarity

df, similarity = load_data()

st.title("🎮 LoL 챔피언 분석 & 추천")

menu = st.sidebar.radio(
    "메뉴 선택",
    ["챔피언 검색", "챔피언 추천"]
)

# -----------------------------
# 챔피언 검색
# -----------------------------
if menu == "챔피언 검색":

    st.header("🔍 챔피언 검색")

    champion = st.selectbox(
        "챔피언 선택",
        sorted(df["챔피언"].tolist())
    )

    info = df[df["챔피언"] == champion].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("기본 정보")

        st.write(f"**챔피언명:** {info['챔피언']}")
        st.write(f"**영문명:** {info['영문이름']}")
        st.write(f"**역할1:** {info['역할1']}")
        st.write(f"**역할2:** {info['역할2']}")

    with col2:
        st.subheader("능력치")

        stats = pd.DataFrame({
            "항목": [
                "체력",
                "마나",
                "방어력",
                "마법저항력",
                "공격력",
                "이동속도",
                "공격사거리"
            ],
            "값": [
                info["체력"],
                info["마나"],
                info["방어력"],
                info["마법저항력"],
                info["공격력"],
                info["이동속도"],
                info["공격사거리"]
            ]
        })

        st.dataframe(stats, use_container_width=True)

# -----------------------------
# 챔피언 추천
# -----------------------------
else:

    st.header("🤖 나와 비슷한 챔피언 추천")

    selected = st.selectbox(
        "좋아하는 챔피언을 선택하세요",
        sorted(df["챔피언"].tolist())
    )

    if st.button("추천 받기"):

        idx = df[df["챔피언"] == selected].index[0]

        scores = list(enumerate(similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        recommendations = []

        for i in scores[1:6]:
            champ = df.iloc[i[0]]
            recommendations.append({
                "챔피언": champ["챔피언"],
                "역할1": champ["역할1"],
                "역할2": champ["역할2"],
                "유사도": round(i[1] * 100, 2)
            })

        rec_df = pd.DataFrame(recommendations)

        st.success(f"{selected}와(과) 비슷한 챔피언 TOP 5")
        st.dataframe(rec_df, use_container_width=True)

    st.divider()

    st.subheader("🎯 플레이 스타일 기반 추천")

    playstyle = st.selectbox(
        "원하는 스타일",
        [
            "탱커",
            "딜러",
            "원거리",
            "근거리",
            "이동속도 빠름"
        ]
    )

    if st.button("스타일 추천"):

        result = df.copy()

        if playstyle == "탱커":
            result = result.sort_values(
                ["체력", "방어력"],
                ascending=False
            )

        elif playstyle == "딜러":
            result = result.sort_values(
                ["공격력", "공격력성장"],
                ascending=False
            )

        elif playstyle == "원거리":
            result = result.sort_values(
                "공격사거리",
                ascending=False
            )

        elif playstyle == "근거리":
            result = result.sort_values(
                "공격사거리",
                ascending=True
            )

        elif playstyle == "이동속도 빠름":
            result = result.sort_values(
                "이동속도",
                ascending=False
            )

        st.dataframe(
            result[
                ["챔피언", "역할1", "역할2",
                 "체력", "공격력", "공격사거리", "이동속도"]
            ].head(10),
            use_container_width=True
        )
