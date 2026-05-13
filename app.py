import streamlit as st
import yfinance as yf
import pandas as pd
from supabase import create_client, Client
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

# 1. Supabase 설정 (Streamlit Secrets에서 관리 권장)
# Streamlit Cloud 배포 시 Settings -> Secrets에 추가하세요.
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

st.set_page_config(page_title="국내 주가 분석기", layout="wide")

st.title("📈 국내 기업 주가 변동 & 뉴스 분석")
st.markdown("종목 코드(6자리)를 입력하면 최신 뉴스 분석과 내일의 예상 주가를 알려줍니다.")

# 2. 사용자 입력
col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input("종목코드 입력 (예: 삼성전자 005930, SK하이닉스 000660)", "005930")
with col2:
    market_type = st.selectbox("시장 구분", ["KOSPI (.KS)", "KOSDAQ (.KQ)"])

suffix = ".KS" if "KOSPI" in market_type else ".KQ"
ticker_symbol = ticker_input + suffix

if st.button("분석 시작"):
    try:
        # 3. 주식 데이터 가져오기
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period="1mo")
        company_name = stock.info.get('longName', ticker_input)

        if df.empty:
            st.error("데이터를 불러오지 못했습니다. 종목코드를 확인해주세요.")
        else:
            # --- 주가 예측 (단순 선형 회귀) ---
            df['Date_n'] = np.arange(len(df))
            X = df[['Date_n']].values
            y = df['Close'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            next_day_index = np.array([[len(df)]])
            predicted_price = model.predict(next_day_index)[0]
            current_price = df['Close'].iloc[-1]
            diff = predicted_price - current_price

            # 결과 출력
            st.subheader(f"📊 {company_name} 분석 결과")
            c1, c2 = st.columns(2)
            c1.metric("현재 주가", f"{int(current_price):,}원")
            c2.metric("내일 예상 주가", f"{int(predicted_price):,}원", f"{int(diff):,}원")

            # --- 뉴스 섹션 ---
            st.divider()
            st.subheader("📰 최신 관련 뉴스")
            news_list = stock.news[:2] # 최신 뉴스 2개

            if news_list:
                for item in news_list:
                    title = item.get('title')
                    link = item.get('link')
                    publisher = item.get('publisher')
                    
                    st.write(f"**[{publisher}]** {title}")
                    st.write(f"🔗 [기사 보기]({link})")
                    
                    # Supabase에 뉴스 저장 (중복 방지 로직은 생략, 필요시 추가 가능)
                    try:
                        supabase.table("stock_news").insert({
                            "company_name": company_name,
                            "title": title,
                            "link": link,
                            "summary": publisher
                        }).execute()
                    except Exception as e:
                        pass # 저장 오류 시 무시
            else:
                st.info("최신 뉴스가 없습니다.")
                
            # 주가 차트
            st.line_chart(df['Close'])

    except Exception as e:
        st.error(f"오류 발생: {e}")

# 하단 안내
st.caption("주의: 위 예측값은 단순 통계 모델에 의한 결과로 투자 책임은 본인에게 있습니다.")
