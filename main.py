import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="LoL Champion Recommender",
    page_icon="🎮",
    layout="wide"
)

# ==========================
# 데이터 로드
# ==========================
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

    return df, similarity, numeric_cols

df, similarity, numeric_cols = load_data()

# ==========================
# 제목
# ==========================
st.title("🎮 League of Legends Champion Analyzer")
st.markdown("챔피언 검색 · 비교 · 추천 서비스")

# ==========================
# 메뉴
# ==========================
menu = st.sidebar.radio(
    "메뉴 선택",
    [
        "챔피언 검색",
        "챔피언 비교",
        "챔피언 추천",
        "플레이 스타일 추천",
        "TOP10 랭킹"
    ]
)

# ===================================================
# 챔피언 검색
# ===================================================
if menu == "챔피언 검색":

    st.header("🔍 챔피언 검색")

    champion = st.selectbox(
        "챔피언 선택",
        sorted(df["챔피언"])
    )

    info = df[df["챔피언"] == champion].iloc[0]

    col1, col2 = st.columns([1,1])

    with col1:

        st.subheader("기본 정보")

        st.write(f"**챔피언** : {info['챔피언']}")
        st.write(f"**영문명** : {info['영문이름']}")
        st.write(f"**역할1** : {info['역할1']}")
        st.write(f"**역할2** : {info['역할2']}")

    with col2:

        st.subheader("능력치")

        st.dataframe(
            pd.DataFrame({
                "항목": numeric_cols,
                "값": [info[col] for col in numeric_cols]
            }),
            use_container_width=True
        )

    # 레이더 차트
    st.subheader("📊 챔피언 능력치 레이더 차트")

    radar_cols = [
        "체력",
        "마나",
        "방어력",
        "마법저항력",
        "공격력",
        "이동속도"
    ]

    scaler = MinMaxScaler()

    radar_values = scaler.fit_transform(
        df[radar_cols]
    )

    idx = df[df["챔피언"] == champion].index[0]

    values = radar_values[idx].tolist()

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=radar_cols,
            fill="toself",
            name=champion
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,1]
            )
        ),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# ===================================================
# 챔피언 비교
# ===================================================
elif menu == "챔피언 비교":

    st.header("⚔️ 챔피언 비교")

    col1, col2 = st.columns(2)

    with col1:
        champ1 = st.selectbox(
            "첫 번째 챔피언",
            sorted(df["챔피언"])
        )

    with col2:
        champ2 = st.selectbox(
            "두 번째 챔피언",
            sorted(df["챔피언"]),
            index=1
        )

    data1 = df[df["챔피언"] == champ1].iloc[0]
    data2 = df[df["챔피언"] == champ2].iloc[0]

    compare = pd.DataFrame({
        champ1: data1[numeric_cols],
        champ2: data2[numeric_cols]
    })

    st.dataframe(compare)

    st.subheader("📈 능력치 비교 그래프")

    st.bar_chart(compare)

# ===================================================
# 챔피언 추천
# ===================================================
elif menu == "챔피언 추천":

    st.header("🤖 비슷한 챔피언 추천")

    selected = st.selectbox(
        "좋아하는 챔피언",
        sorted(df["챔피언"])
    )

    if st.button("추천 받기"):

        idx = df[df["챔피언"] == selected].index[0]

        scores = list(enumerate(similarity[idx]))

        scores = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )

        recommendations = []

        for i in scores[1:6]:

            champ = df.iloc[i[0]]

            recommendations.append({
                "챔피언": champ["챔피언"],
                "유사도": round(i[1] * 100, 2),
                "역할1": champ["역할1"]
            })

        rec_df = pd.DataFrame(recommendations)

        st.dataframe(rec_df)

        st.subheader("📊 추천 챔피언 유사도")

        fig = px.bar(
            rec_df,
            x="챔피언",
            y="유사도",
            text="유사도"
        )

        st.plotly_chart(fig, use_container_width=True)

# ===================================================
# 플레이 스타일 추천
# ===================================================
elif menu == "플레이 스타일 추천":

    st.header("🎯 플레이 스타일 추천")

    hp_weight = st.slider("체력 중요도", 0, 100, 50)
    attack_weight = st.slider("공격력 중요도", 0, 100, 50)
    speed_weight = st.slider("이동속도 중요도", 0, 100, 50)

    score = (
        (df["체력"] / df["체력"].max()) * hp_weight +
        (df["공격력"] / df["공격력"].max()) * attack_weight +
        (df["이동속도"] / df["이동속도"].max()) * speed_weight
    )

    result = df.copy()

    result["추천점수"] = score

    result = result.sort_values(
        "추천점수",
        ascending=False
    )

    st.dataframe(
        result[
            ["챔피언", "역할1", "역할2", "추천점수"]
        ].head(10)
    )

# ===================================================
# TOP10 랭킹
# ===================================================
elif menu == "TOP10 랭킹":

    st.header("🏆 공격력 TOP10")

    top10 = (
        df.sort_values(
            "공격력",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        top10,
        x="챔피언",
        y="공격력",
        text="공격력"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        top10[
            ["챔피언", "공격력", "역할1"]
        ]
    )
