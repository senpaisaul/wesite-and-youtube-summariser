import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.docstore.document import Document
from youtube_transcript_api import YouTubeTranscriptApi
import re
import urllib.parse

## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## Gemma Model Using Groq API
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def extract_youtube_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and shortened URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embedded URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # youtu.be URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def get_youtube_transcript(video_id):
    """Get transcript from YouTube video using youtube-transcript-api"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript_list])
    except Exception as e:
        st.error(f"Error fetching YouTube transcript: {str(e)}")
        return None

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or website URL")
    else:
        try:
            with st.spinner("Fetching content..."):
                docs = []
                
                # Handle YouTube URLs
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    video_id = extract_youtube_id(generic_url)
                    
                    if not video_id:
                        st.error("Could not extract YouTube video ID from the URL")
                    else:
                        st.info(f"Processing YouTube video ID: {video_id}")
                        transcript = get_youtube_transcript(video_id)
                        
                        if transcript:
                            # Create a Document object from the transcript
                            docs = [Document(page_content=transcript, metadata={"source": generic_url})]
                        else:
                            st.error("Could not retrieve transcript for this video. It may not have subtitles available.")
                
                # Handle regular website URLs
                else:
                    st.info(f"Processing website: {generic_url}")
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                    docs = loader.load()
                
                if not docs:
                    st.error("Could not extract content from the provided URL. Please try another URL.")
                else:
                    st.info("Content extracted successfully! Generating summary...")
                    
                    # Chain For Summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)
                    
                    st.success("Summary:")
                    st.write(output_summary)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
