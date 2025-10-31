import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import joblib
# -------------------- 数据加载与初始化 --------------------
@st.cache_data
def load_data():
    df = pd.read_csv('student_data.csv',encoding='UTF-8') # 读取数据文件路径
    return df
df = load_data()
# 初始化模型（若需训练新模型，可取消注释下方训练代码）
# def train_model():
# X = df[["每周学习时长", "上课出勤率", "期中考试分数", "作业完成率"]]
# y = df["期末考试分数"]
# model = LinearRegression()
# model.fit(X, y)
# joblib.dump(model, "score_predictor.pkl")
# train_model() # 首次运行时训练模型，之后可注释
model = joblib.load("score_predictor.pkl") # 加载训练好的模型




import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
st.set_page_config(page_title='学生成绩分析与预测系统',page_icon='🎓',layout='wide')

with st.sidebar:
    page = st.radio(
        "🎓导航菜单",
        ("项目介绍", "专业数据分析", "成绩预测")
    )
#1
if page == "项目介绍":
    st.title('📝学生成绩分析与系统预测')
    a1,a2 = st.columns(2)
    with a1:
        st.markdown('***')
        st.header("📔项目概述")
        st.text('''本项目是一个基于Streamlit的学生成绩分析平台，通过数据可视化和机器学习技术，帮助教育工作者和学生深入了解学业表现，并预测期末考试成绩。''')
        st.subheader("主要特点:")
        st.markdown('''- **🔍数据可视化**:  多维度膳食学生学业数据
- **📃专业分析**:  按专业分类的详细统计分析
- **🏷智能预测**:  基于机器学习模型的成绩预测
- **💡学习建议**:  根据预测结果提供个性化反馈''')
    with a2:
        st.image('page2.png', width=600)
        st.text('学生数据分析示意图')
    st.markdown('***')
    st.header("🚀项目目标")
    b1,b2,b3 = st.columns(3)
    with b1:
        st.subheader("🌊目标一")
        st.text('⌛分析影响因素')
        st.markdown('''- 识别关键学习指标
    - 探索成绩相关因素
    - 提供数据支持决策''')
    with b2:
        st.subheader("🛎目标二")
        st.text('🪐可视化展示')
        st.markdown('''- 专业对比分析
    - 性别研究差异
    - 学习模式识别''')
    with b3:
        st.subheader("🛶目标三")
        st.text('🌌成绩预测')
        st.markdown('''- 机器学习模型
    - 个性化预测
    - 及时干预预警''')
    st.markdown('***')
    st.header("🚀技术架构")
    c1,c2,c3,c4 = st.columns(4)
    with c1:  
        st.text('前端框架')
        python_code = '''Streamlit'''
        st.code(python_code,language='python',line_numbers=True)
    with c2:  
        st.text('数据处理')
        python_code = '''Pandas
    Numpy'''
        st.code(python_code,language='python',line_numbers=True)
    with c3:  
        st.text('可视化')
        python_code = '''Plotly
    Natplotlib'''
        st.code(python_code,language='python',line_numbers=True)
    with c4:  
        st.text('机器学习')
        python_code = '''Scikit-learn'''
        st.code(python_code,language='python',line_numbers=True)
#2
elif page == "专业数据分析":
    import plotly.graph_objects as go
    import pandas as pd
    import plotly.express as px
    import altair as alt
    st.header("📊专业数据分析")
    st.markdown('***')
    st.subheader("1. 各专业男女性别比例")
    df_student = pd.read_csv("student_data.csv")
    gender_count = df_student.groupby(["专业", "性别"]).size().unstack(fill_value=0)
    if gender_count.columns.tolist() == ["男", "女"]:
        gender_count = gender_count[["女", "男"]]
    gender_ratio = (gender_count / gender_count.sum(axis=1).values.reshape(-1, 1) * 100).round(1)
    df_gender = gender_ratio.reset_index()
    df_gender.columns = ["major", "女", "男"]  
    fig_gender = go.Figure()
    fig_gender.add_trace(go.Bar(
        x=df_gender["major"],
        y=df_gender["男"],
        name="男",
        marker_color="#87CEEB"
    ))
    fig_gender.add_trace(go.Bar(
        x=df_gender["major"],
        y=df_gender["女"],
        name="女",
        marker_color="#4169E1"
    ))
    fig_gender.update_layout(
        barmode="group",  
        xaxis_title="专业",  
        yaxis_title="比例(%)", 
        height=400,  
        legend_title="性别",  
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99)  
    )
    col1, col2 = st.columns([2, 1]) 
    with col1:       
        st.plotly_chart(fig_gender, use_container_width=True)
    with col2:       
        st.subheader("性别比例数据")       
        st.dataframe(df_gender.set_index("major"), use_container_width=True)
    st.markdown('***')
    st.header("2.各专业学习指标对比")
    st.caption("各专业平均学习时间与成绩对比")
    df = pd.read_csv("student_data.csv")
    metrics = ["每周学习时长（小时）", "期中考试分数", "期末考试分数"]
    df_major = df.groupby("专业")[metrics].mean().round(1).reset_index()
    df_melt = df_major.melt(id_vars="专业", var_name="指标", value_name="时间")
    bar_layer = alt.Chart(df_melt[df_melt["指标"] == "期中考试分数"]).mark_bar(color="#4169E1").encode(
        x=alt.X("专业", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("时间", title="平均学习时间"),
        tooltip=["专业", "指标", "时间"]
    )
    line_layer1 = alt.Chart(df_melt[df_melt["指标"] == "每周学习时长（小时）"]).mark_line(point=True, color="#87CEEB").encode(
        x="专业",
        y="时间",
        tooltip=["专业", "指标", "时间"]
    )

    line_layer2 = alt.Chart(df_melt[df_melt["指标"] == "期末考试分数"]).mark_line(point=True, color="#FF6347").encode(
        x="专业",
        y="时间",
        tooltip=["专业", "指标", "时间"]
    )
    chart = bar_layer + line_layer1 + line_layer2
    chart = chart.properties(height=400).configure_axis(titleFontSize=14, labelFontSize=12)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.altair_chart(chart, use_container_width=True, theme="streamlit")
    with col2:
        st.subheader("详细数据")
        st.dataframe(df_major.set_index("专业"), use_container_width=True)
    st.markdown('***')
    st.header("3.各专业出勤率分析")
    st.subheader("出勤率排名")
    df = pd.read_csv("student_data.csv")
    df_attendance = df.groupby("专业")["上课出勤率"].mean().reset_index()
    df_attendance["平均出勤率"] = (df_attendance["上课出勤率"] * 100).round(1)
    df_attendance_sorted = df_attendance.sort_values("平均出勤率", ascending=False).reset_index(drop=True)
    df_attendance_sorted["排名"] = df_attendance_sorted.index
    df_result = df_attendance_sorted[["排名", "专业", "平均出勤率"]]

    chart = (
        alt.Chart(df_result)
        .mark_bar()
        .encode(
            x=alt.X("专业", sort="-y", title="专业"),
            y=alt.Y("平均出勤率", title="平均出勤率(%)"),
            color=alt.Color("专业", scale=alt.Scale(scheme="category10")),  
            tooltip=["排名", "专业", "平均出勤率"]
        )
        .properties(height=350)
    )
    col_chart, col_table = st.columns([1, 1])
    with col_chart:
        st.altair_chart(chart, use_container_width=True, theme="streamlit")
    with col_table:
        st.dataframe(df_result.set_index("排名"), use_container_width=True)

    st.markdown('***')
    st.header("4.大数据管理专业专项分析")

    df = pd.read_csv("student_data.csv")
    df_bd = df[df["专业"] == "大数据管理"].copy()

    avg_study = df_bd["每周学习时长（小时）"].mean().round(1)
    avg_attend = (df_bd["上课出勤率"].mean() * 100).round(1)
    avg_final = df_bd["期末考试分数"].mean().round(1)
    pass_rate = round((len(df_bd[df_bd["期末考试分数"] >= 60]) / len(df_bd) * 100), 1)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均学习时间", f"{avg_study} 小时")
    with col2:
        st.metric("平均出勤率", f"{avg_attend}%")
    with col3:
        st.metric("平均期末成绩", f"{avg_final} 分")
    with col4:
        st.metric("及格率", f"{pass_rate}%")

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("大数据管理专业期末成绩分布")
        histogram = (
            alt.Chart(df_bd)
            .mark_bar(color="#87CEEB", opacity=0.8)
            .encode(
                x=alt.X("期末考试分数:Q", title="期末成绩（分）", bin=alt.Bin(maxbins=10)),
                y=alt.Y("count():Q", title="学生人数"),
                tooltip=[alt.Tooltip("期末考试分数:Q", bin=True, title="成绩区间"), "count():Q"]
            )
            .properties(height=350)
        )
        st.altair_chart(histogram, use_container_width=True, theme="streamlit")
            
    with col6:
        st.subheader("大数据管理专业期末成绩箱线图")
        boxplot = (
            alt.Chart(df_bd)
            .mark_boxplot(color="#4169E1", size=70)
            .encode(
                x=alt.X("专业:N", title="专业", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("期末考试分数:Q", title="期末成绩（分）", scale=alt.Scale(domain=[40, 100])),
                tooltip=["期末考试分数"]
            )
            .properties(height=360)
        )
        st.altair_chart(boxplot, use_container_width=True, theme="streamlit")
  
#3    
else:
    st.subheader("〽️期末成绩预测")
    st.markdown('***')
    df = pd.read_csv("student_data.csv")
    st.text_area(label='', placeholder='请输入学生的学习信息，系统将预测其期末成绩并提供学习建议')
    with st.form("predict_form"):
        st.subheader("请输入学生信息")    
    
        student_id = st.text_input("学号")
        gender = st.selectbox("性别", ["男", "女"])
        major = st.selectbox("专业", df["专业"].unique())
        study_hours = st.number_input("每周学习时长（小时）", min_value=0.0, max_value=50.0, step=0.1)
        attendance = st.number_input("上课出勤率", min_value=0.0, max_value=1.0, step=0.01)
        mid_score = st.number_input("期中考试分数", min_value=0.0, max_value=100.0, step=0.1)
        homework_rate = st.number_input("作业完成率", min_value=0.0, max_value=1.0, step=0.01)
        submit = st.form_submit_button("预测成绩")

    if submit:

        X = [[study_hours, attendance, mid_score, homework_rate]]
        pred_score = model.predict(X)[0]
        pred_score = max(0, min(100, pred_score)) 
        st.subheader("预测结果")
        st.markdown(f"**预测期末成绩：{pred_score:.2f} 分**")
        if pred_score >= 80:
            st.image("1.jpg") 
        elif pred_score >= 60:
            st.success("成绩合格，继续保持！")
            st.image('2.jpg')
        else:
            st.warning("成绩待提高，建议加强学习！")
            st.image('3.jpg')

