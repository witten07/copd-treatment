import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import streamlit as st
import re

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

def get_copd_trials(max_results=10):
    url = f"https://clinicaltrials.gov/ct2/results/rss.xml?cond=COPD&count={max_results}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'xml')
    items = soup.find_all('item')
    trials = []
    for item in items:
        title = item.title.text
        link = item.link.text
        summary = re.sub('<[^<]+?>', '', item.description.text)
        trials.append({"title": title, "link": link, "summary": summary})
    return trials

def summarize_trials(trials):
    for trial in trials:
        try:
            result = summarizer(trial["summary"], max_length=100, min_length=30, do_sample=False)
            trial["summary_short"] = result[0]['summary_text']
        except Exception as e:
            trial["summary_short"] = f"Could not summarize: {str(e)}"
    return trials

def filter_trials(trials, keyword):
    if not keyword:
        return trials
    return [t for t in trials if keyword.lower() in t["summary"].lower() or keyword.lower() in t["title"].lower()]

def main():
    st.set_page_config(page_title="COPD AI Research Assistant", layout="centered")
    st.title("COPD Treatment Finder")
    st.caption("AI-powered tool to summarize and explore the latest COPD clinical trials")

    max_trials = st.slider("How many trials to fetch?", 1, 30, 10)
    keyword = st.text_input("Optional keyword to filter (e.g. 'stem cell', 'biologic', 'inhaled')")

    if st.button("Search Trials"):
        with st.spinner("Fetching and analyzing trials..."):
            trials = get_copd_trials(max_trials)
            trials = filter_trials(trials, keyword)
            trials = summarize_trials(trials)

        if not trials:
            st.warning("No trials found for your criteria.")
        for trial in trials:
            st.subheader(trial['title'])
            st.markdown(f"[View Full Trial]({trial['link']})")
            st.write(trial['summary_short'])
            st.divider()

if __name__ == "__main__":
    main()
