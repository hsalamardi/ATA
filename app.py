import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertForTokenClassification,
    pipeline
)
import numpy as np
import re
from torch.nn.functional import softmax
from typing import List, Dict
import plotly.graph_objects as go
import torch.cuda
import gc
from functools import lru_cache
import logging
import os
from pathlib import Path
import io
from model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="محلل النصوص العربية",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        font-family: 'Arial', sans-serif;
    }
    .main {
        padding: 2rem;
    }
    .st-emotion-cache-1v0mbdj {
        width: 100%;
    }
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        background-color: #B6FFB6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .sentiment-neutral {
        background-color: #CCCCFF;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .sentiment-negative {
        background-color: #FFB6B6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .entity-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        margin: 0.25rem;
        font-size: 0.9em;
    }
    .download-button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        margin-top: 1rem;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

class ArabicPreprocessor:
    """Arabic text preprocessing class with comprehensive cleaning capabilities."""
    def __init__(self):
        self.arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670]')
        self.arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ'''
        self.arabic_numbers = re.compile(r'[٠١٢٣٤٥٦٧٨٩]')
        self.english_numbers = re.compile(r'[0-9]')
        self.hashtags = re.compile(r'#\w+')
        self.mentions = re.compile(r'@\w+')
        self.urls = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.emails = re.compile(r'\S+@\S+')
        
    def preprocess(self, text: str) -> str:
        """Preprocess Arabic text with comprehensive cleaning."""
        if not isinstance(text, str) or not text:
            return ""

        try:
            # Remove URLs and emails
            text = self.urls.sub(' ', text)
            text = self.emails.sub(' ', text)
            
            # Remove diacritics and normalize Arabic letters
            text = self.arabic_diacritics.sub('', text)
            text = re.sub('[إأآا]', 'ا', text)
            text = re.sub('ة', 'ه', text)
            text = re.sub('[ىي]', 'ي', text)
            
            # Normalize Arabic numbers
            text = self.arabic_numbers.sub(
                lambda m: str(int(m.group())), 
                text
            )
            
            # Remove punctuations
            text = text.translate(str.maketrans('', '', self.arabic_punctuations))
            
            # Remove social media markers
            text = self.hashtags.sub(' ', text)
            text = self.mentions.sub(' ', text)
            
            # Remove repeated characters (more than 2)
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)
            
            # Clean up whitespace
            text = ' '.join(text.split())
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return text  # Return original text if preprocessing fails

class ArabicAnalyzer:
    def __init__(self, cache_dir: str = None):
        self.preprocessor = ArabicPreprocessor()
        self.cache_dir = cache_dir or os.path.join(Path.home(), '.cache/arabic_analyzer')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Define models
        self.models_config = {
            'sentiment': {
                'path': 'CAMeL-Lab/bert-base-arabic-camelbert-mix',
                        #CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment
                'tokenizer': BertTokenizer,
                'model': BertForSequenceClassification,
                'max_length': 512,
                'num_labels': 3,
                'id2label': {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
            },
            'classification': {
                'path': 'UBC-NLP/MARBERT',
                'tokenizer': BertTokenizer,
                'model': BertForSequenceClassification,
                'max_length': 512,
                'num_labels': 8,
                'id2label': {
                    0: "سياسة",
                    1: "رياضة",
                    2: "اقتصاد",
                    3: "تكنولوجيا",
                    4: "ثقافة",
                    5: "صحة",
                    6: "تعليم",
                    7: "ترفيه"
                }
            },
            'ner': {
                'path': 'aubmindlab/bert-base-arabertv02',
                'tokenizer': BertTokenizer,
                'model': BertForTokenClassification,
                'max_length': 256,
                'num_labels': 9,
                'id2label': {
                    0: "O",
                    1: "B-PERS",
                    2: "I-PERS",
                    3: "B-ORG",
                    4: "I-ORG",
                    5: "B-LOC",
                    6: "I-LOC",
                    7: "B-DATE",
                    8: "I-DATE"
                }
            }
        }
        
        # Topic categories
        self.topic_categories = self.models_config['classification']['id2label']
        
        # NER labels
        self.ner_labels = {
            'B-PERS': 'شخص',
            'I-PERS': 'شخص',
            'B-ORG': 'منظمة',
            'I-ORG': 'منظمة',
            'B-LOC': 'مكان',
            'I-LOC': 'مكان',
            'B-DATE': 'تاريخ',
            'I-DATE': 'تاريخ',
            'O': ''
        }
        
        # Initialize model cache
        self._model_cache = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.load_models()

    def __del__(self):
        """Cleanup when the analyzer is destroyed."""
        self._clear_gpu_memory()
        for cache_item in self._model_cache.values():
            if 'pipeline' in cache_item:
                del cache_item['pipeline']
        self._model_cache.clear()

    def _clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _load_model(self, task: str, config: dict):
        """Load a single model with proper configuration."""
        try:
            cache_path = os.path.join(self.cache_dir, task)
            
            if os.path.exists(cache_path):
                tokenizer = config['tokenizer'].from_pretrained(cache_path)
                model = config['model'].from_pretrained(
                    cache_path,
                    num_labels=config['num_labels'],
                    id2label=config['id2label'],
                    label2id={v: k for k, v in config['id2label'].items()}
                )
                logger.info(f"Loaded {task} model from cache")
            else:
                tokenizer = config['tokenizer'].from_pretrained(config['path'])
                model = config['model'].from_pretrained(
                    config['path'],
                    num_labels=config['num_labels'],
                    id2label=config['id2label'],
                    label2id={v: k for k, v in config['id2label'].items()}
                )
                tokenizer.save_pretrained(cache_path)
                model.save_pretrained(cache_path)
                logger.info(f"Saved {task} model to cache")
            
            model.to(self.device)
            pipe_task = "sentiment-analysis" if task == 'sentiment' else \
                       "text-classification" if task == 'classification' else \
                       "token-classification"
            
            self._model_cache[task] = {
                'pipeline': pipeline(
                    pipe_task,
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                ),
                'max_length': config['max_length'],
                'id2label': config['id2label']
            }
            
            st.success(f"✅ تم تحميل نموذج {task} بنجاح")
            
        except Exception as e:
            logger.error(f"Error loading {task} model: {str(e)}")
            st.error(f"❌ خطأ في تحميل نموذج {task}: {str(e)}")

    def load_models(self):
        """Load all models with proper error handling."""
        try:
            for task, config in self.models_config.items():
                self._load_model(task, config)
        except Exception as e:
            logger.error(f"Error in load_models: {str(e)}")
            st.error(f"❌ حدث خطأ أثناء تحميل النماذج: {str(e)}")
            raise e

    def _analyze_text_internal(self, text: str) -> dict:
        """Internal method for text analysis with improved error handling."""
        results = {}
        
        try:
            if not text or not isinstance(text, str):
                raise ValueError("النص المدخل غير صالح")
            
            processed_text = self.preprocessor.preprocess(text)
            
            # Sentiment Analysis
            if 'sentiment' in self._model_cache:
                try:
                    sentiment_result = self._model_cache['sentiment']['pipeline'](processed_text)
                    label = sentiment_result[0]['label']
                    results['sentiment'] = {
                        'label': 'إيجابي' if label == 'POSITIVE' else 'محايد' if label == 'NEUTRAL' else 'سلبي',
                        'score': sentiment_result[0]['score']
                    }
                except Exception as e:
                    logger.error(f"Error in sentiment analysis: {e}")
                    results['sentiment'] = {'error': str(e)}

            # Topic Classification
            if 'classification' in self._model_cache:
                try:
                    class_result = self._model_cache['classification']['pipeline'](processed_text)
                    # Create reverse mapping from label to id
                    label2id = {v: k for k, v in self._model_cache['classification']['id2label'].items()}
                    label = class_result[0]['label']
                    # If the label is in LABEL_X format, extract X, otherwise use the mapping
                    if label.startswith('LABEL_'):
                        label_id = int(label.split('_')[1])
                    else:
                        label_id = label2id.get(label, 0)
                    
                    results['topic'] = {
                        'category': self._model_cache['classification']['id2label'][label_id],
                        'score': class_result[0]['score']
                    }
                except Exception as e:
                    logger.error(f"Error in classification: {e}")
                    results['topic'] = {'error': str(e)}

            # Named Entity Recognition
            if 'ner' in self._model_cache:
                try:
                    ner_results = self._model_cache['ner']['pipeline'](processed_text)
                    entities = []
                    # Create reverse mapping from label to id
                    label2id = {v: k for k, v in self._model_cache['ner']['id2label'].items()}
                    
                    for ent in ner_results:
                        entity = ent['entity']
                        if entity == 'O':
                            continue
                            
                        # Handle both LABEL_X format and direct label format
                        if entity.startswith('LABEL_'):
                            label_id = int(entity.split('_')[1])
                            entity_type = self._model_cache['ner']['id2label'][label_id]
                        else:
                            entity_type = entity
                            
                        if entity_type in self.ner_labels:
                            entities.append({
                                'text': ent['word'],
                                'type': self.ner_labels[entity_type],
                                'score': ent['score']
                            })
                    results['entities'] = entities
                except Exception as e:
                    logger.error(f"Error in NER: {e}")
                    results['entities'] = {'error': str(e)}
            
            results['text'] = text
            return results
            
        except Exception as e:
            logger.error(f"Error in text analysis: {str(e)}")
            return {'error': str(e)}
        finally:
            self._clear_gpu_memory()

    @lru_cache(maxsize=1000)
    def _cached_analysis(self, text: str) -> dict:
        """Cached version of text analysis to avoid recomputing frequent texts."""
        return self._analyze_text_internal(text)

    def analyze_text(self, text: str) -> Dict:
        """Public method for text analysis with caching."""
        try:
            # Check if input is valid
            if not isinstance(text, str) or not text.strip():
                return {
                    'error': 'النص فارغ أو غير صالح',
                    'text': text
                }

            # Use cache if available
            cached_result = self._cached_analysis(text)
            if cached_result.get('error'):
                # If there was an error, try to recover by reloading models
                logger.warning("Analysis failed, attempting recovery...")
                self.load_models()
                cached_result = self._cached_analysis(text)
            
            return cached_result
            
        except Exception as e:
            logger.error(f"Critical error in analyze_text: {str(e)}")
            return {
                'error': str(e),
                'text': text
            }
        finally:
            self._clear_gpu_memory()

    def analyze_batch(self, texts: List[str], progress_callback=None) -> List[Dict]:
        """Process multiple texts with memory optimization."""
        results = []
        batch_size = 10  # Process 10 texts at a time
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                result = self.analyze_text(text)
                batch_results.append(result)
                
                if progress_callback:
                    progress = (i + len(batch_results)) / len(texts)
                    progress_callback(progress)
            
            results.extend(batch_results)
            self._clear_gpu_memory()
        
        return results

def create_visualizations(df: pd.DataFrame):
    """Create visualization charts for analysis results."""
    try:
        fig_sentiment = fig_topic = fig_entities = None
        
        # Create sentiment distribution chart
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].apply(lambda x: x['label'] if isinstance(x, dict) else None).value_counts()
            fig_sentiment = go.Figure(data=[
                go.Bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    text=sentiment_counts.values,
                    textposition='auto',
                    marker_color=['#FF9999', '#CCCCFF', '#66B2FF']
                )
            ])
            fig_sentiment.update_layout(
                title={
                    'text': "توزيع المشاعر",
                    'x': 1,
                    'xanchor': 'right'
                },
                xaxis_title="التصنيف",
                yaxis_title="العدد",
                xaxis={'side': 'top'},
                font={'family': "Arial, sans-serif"},
                hoverlabel={'align': 'right'}
            )
        
        # Create topic distribution chart
        if 'topic' in df.columns:
            topic_counts = df['topic'].apply(lambda x: x['category'] if isinstance(x, dict) else None).value_counts()
            fig_topic = go.Figure(data=[
                go.Bar(
                    x=topic_counts.index,
                    y=topic_counts.values,
                    text=topic_counts.values,
                    textposition='auto',
                    marker_color='#99FF99'
                )
            ])
            fig_topic.update_layout(
                title={
                    'text': "توزيع المواضيع",
                    'x': 1,
                    'xanchor': 'right'
                },
                xaxis_title="الموضوع",
                yaxis_title="العدد",
                xaxis={'side': 'top'},
                font={'family': "Arial, sans-serif"},
                hoverlabel={'align': 'right'}
            )
        
        # Create entity type distribution chart
        if 'entities' in df.columns:
            all_entities = []
            for entities in df['entities']:
                if isinstance(entities, list):
                    all_entities.extend([e['type'] for e in entities if isinstance(e, dict) and 'type' in e])
            
            if all_entities:
                entity_counts = pd.Series(all_entities).value_counts()
                fig_entities = go.Figure(data=[
                    go.Pie(
                        labels=entity_counts.index,
                        values=entity_counts.values,
                        hole=.3
                    )
                ])
                fig_entities.update_layout(
                    title={
                        'text': "توزيع أنواع الكيانات",
                        'x': 1,
                        'xanchor': 'right'
                    },
                    font={'family': "Arial, sans-serif"},
                    hoverlabel={'align': 'right'}
                )
        
        return fig_sentiment, fig_topic, fig_entities
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        st.error(f"خطأ في إنشاء الرسوم البيانية: {str(e)}")
        return None, None, None

def display_entity_details(entity: Dict):
    """Display entity details with color-coded styling."""
    color_map = {
        'شخص': '#FFB6C6',
        'منظمة': '#B6FFB6',
        'مكان': '#B6B6FF',
        'تاريخ': '#FFE4B6'
    }
    background_color = color_map.get(entity['type'], '#F0F0F0')
    
    st.markdown(f"""
        <div style='
            background-color: {background_color};
            padding: 10px;
            border-radius: 5px;
            margin: 5px;
            text-align: right;
            direction: rtl;
        '>
            <p style='font-weight: bold; margin: 0;'>{entity['text']}</p>
            <p style='margin: 0;'>النوع: {entity['type']}</p>
            <div style='
                background-color: white;
                border-radius: 3px;
                margin-top: 5px;
            '>
                <div style='
                    background-color: {background_color};
                    width: {entity['score']*100}%;
                    height: 10px;
                    border-radius: 3px;
                '></div>
            </div>
            <p style='margin: 0; font-size: 0.8em;'>الثقة: {entity['score']:.2%}</p>
        </div>
    """, unsafe_allow_html=True)

def get_entity_color(entity_type):
    """Return color for entity type."""
    colors = {
        'شخص': '#FFB6C1',  # Light pink
        'منظمة': '#98FB98',  # Pale green
        'مكان': '#87CEEB',  # Sky blue
        'تاريخ': '#DDA0DD'   # Plum
    }
    return colors.get(entity_type, '#F0E68C')  # Default to khaki

def process_single_text(text: str, show_scores: bool):
    """Process single text analysis."""
    with st.spinner("جاري تحليل النص..."):
        analyzer = ArabicAnalyzer()
        analysis = analyzer.analyze_text(text)
        
        if 'error' not in analysis:
            display_analysis_results(analysis, text, show_scores)
        else:
            st.error(f"حدث خطأ أثناء التحليل: {analysis.get('error', 'خطأ غير معروف')}")

def process_batch_analysis(df: pd.DataFrame):
    """Process batch analysis of CSV file."""
    try:
        analyzer = ArabicAnalyzer()
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize results storage
        all_results = []
        total = len(df)
        
        # Process each text
        for idx, row in df.iterrows():
            status_text.text(f"جاري تحليل النص {idx + 1} من {total}...")
            progress_bar.progress((idx + 1) / total)
            
            try:
                analysis = analyzer.analyze_text(row['text'])
                if 'error' not in analysis:
                    result = {
                        'النص': row['text'],
                        'المشاعر': analysis['sentiment']['label'],
                        'درجة الثقة (المشاعر)': f"{analysis['sentiment']['score']:.2%}",
                        'التصنيف': analysis['topic']['category'],
                        'درجة الثقة (التصنيف)': f"{analysis['topic']['score']:.2%}",
                        'الكيانات': ' | '.join([f"{e['text']} ({e['type']})" for e in analysis.get('entities', [])]),
                        'درجات الثقة (الكيانات)': ' | '.join([f"{e['score']:.2%}" for e in analysis.get('entities', [])])
                    }
                else:
                    result = {
                        'النص': row['text'],
                        'خطأ': analysis.get('error', 'خطأ غير معروف')
                    }
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing text {idx}: {e}")
                all_results.append({
                    'النص': row['text'],
                    'خطأ': str(e)
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Display summary
        st.success("✅ تم اكتمال التحليل!")
        st.markdown("### 📊 ملخص النتائج")
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("إجمالي النصوص", total)
        with col2:
            successful = len(results_df[~results_df['المشاعر'].isna()]) if 'المشاعر' in results_df.columns else 0
            st.metric("النصوص الناجحة", successful)
        with col3:
            failed = total - successful
            st.metric("النصوص الفاشلة", failed)
        
        # Show results table
        st.dataframe(results_df, use_container_width=True)
        
        # Create Excel file
        excel_buffer = io.BytesIO()
        results_df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_data = excel_buffer.getvalue()
        
        # Add download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="⬇️ تحميل النتائج (Excel)",
                data=excel_data,
                file_name="نتائج_التحليل.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with col2:
            st.download_button(
                label="⬇️ تحميل النتائج (CSV)",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name="نتائج_التحليل.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تحليل الملف: {str(e)}")
        logger.error(f"Batch processing error: {e}")

def display_analysis_results(analysis: dict, text: str, show_scores: bool):
    """Display analysis results with modern UI."""
    # Create DataFrame for export
    results_data = {
        'النص': [text],
        'المشاعر': [analysis['sentiment']['label']],
        'درجة الثقة (المشاعر)': [f"{analysis['sentiment']['score']:.2%}"],
        'التصنيف': [analysis['topic']['category']],
        'درجة الثقة (التصنيف)': [f"{analysis['topic']['score']:.2%}"],
    }
    
    # Add entities to the DataFrame
    entities_text = []
    entities_types = []
    entities_scores = []
    for entity in analysis.get('entities', []):
        entities_text.append(entity['text'])
        entities_types.append(entity['type'])
        entities_scores.append(f"{entity['score']:.2%}")
    
    results_data['الكيانات'] = [' | '.join(entities_text)]
    results_data['أنواع الكيانات'] = [' | '.join(entities_types)]
    results_data['درجات الثقة (الكيانات)'] = [' | '.join(entities_scores)]
    
    df = pd.DataFrame(results_data)
    
    # Convert DataFrame to Excel
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_data = excel_buffer.getvalue()
    
    st.markdown("### 📊 نتائج التحليل")
    
    # Display results in a modern UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='result-card'>
                <h4>تحليل المشاعر</h4>
                <div class='sentiment-{}'>{}</div>
                {}
            </div>
        """.format(
            'positive' if analysis['sentiment']['label'] == 'إيجابي' 
            else 'neutral' if analysis['sentiment']['label'] == 'محايد' 
            else 'negative',
            analysis['sentiment']['label'],
            f"<p>درجة الثقة: {analysis['sentiment']['score']:.2%}</p>" if show_scores else ""
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='result-card'>
                <h4>تصنيف الموضوع</h4>
                <div style='text-align: center; font-size: 1.2em;'>{}</div>
                {}
            </div>
        """.format(
            analysis['topic']['category'],
            f"<p>درجة الثقة: {analysis['topic']['score']:.2%}</p>" if show_scores else ""
        ), unsafe_allow_html=True)
    
    st.markdown("""
        <div class='result-card'>
            <h4>الكيانات المستخرجة</h4>
            <div style='margin-top: 1rem;'>
    """, unsafe_allow_html=True)
    
    for entity in analysis.get('entities', []):
        st.markdown(f"""
            <span class='entity-tag' style='background-color: {get_entity_color(entity["type"])}'>
                {entity['text']} ({entity['type']})
                {f"<br><small>{entity['score']:.2%}</small>" if show_scores else ""}
            </span>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Add download button for Excel
    st.download_button(
        label="⬇️ تحميل النتائج (Excel)",
        data=excel_data,
        file_name="نتائج_التحليل.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def main():
    """Main application function."""
    trainer = ModelTrainer()
    current_version = trainer.get_current_version()
    
    tab1, tab2, tab3 = st.tabs(["📝 تحليل نص", "📊 تحليل ملف CSV", "🎯 تدريب النموذج"])
    
    with tab1:
        st.title("🔍 محلل النصوص العربية")
        st.markdown(f"إصدار النموذج الحالي: {current_version}")
        st.markdown("---")

        with st.sidebar:
            st.header("⚙️ الإعدادات")
            max_length = st.slider("الحد الأقصى لطول النص", 100, 1000, 500)
            show_scores = st.checkbox("عرض درجات الثقة", value=True)
            
        text_input = st.text_area(
            "أدخل النص العربي هنا",
            height=200,
            placeholder="اكتب أو الصق النص العربي هنا للتحليل..."
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            analyze_button = st.button("تحليل النص", type="primary")
        with col2:
            clear_button = st.button("مسح النص", type="secondary")

        if clear_button:
            st.session_state.text_input = ""
            st.session_state.analysis_results = None
            st.rerun()

        if analyze_button and text_input:
            process_single_text(text_input, show_scores)

    with tab2:
        st.title("📊 تحليل ملف CSV")
        st.markdown(f"إصدار النموذج الحالي: {current_version}")
        st.markdown("---")
        
        st.info("""
        قم بتحميل ملف CSV يحتوي على عمود 'text' للتحليل. 
        يجب أن يكون الملف بتنسيق UTF-8 ويحتوي على عمود باسم 'text'.
        """)
        
        uploaded_file = st.file_uploader(
            "قم بتحميل ملف CSV:",
            type=['csv'],
            help="يجب أن يحتوي الملف على عمود 'text'"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("⚠️ يجب أن يحتوي الملف على عمود 'text'")
                    return
                
                st.success(f"✅ تم تحميل الملف بنجاح. عدد النصوص: {len(df)}")
                
                if st.button("🚀 تحليل الملف", type="primary"):
                    process_batch_analysis(df)
                    
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء قراءة الملف: {str(e)}")

    with tab3:
        st.title("🎯 تدريب النموذج")
        st.markdown(f"إصدار النموذج الحالي: {current_version}")
        st.markdown("---")
        
        st.info("""
        قم بتحميل ملف CSV للتدريب. يجب أن يحتوي الملف على العمودين التاليين:
        - text: النص العربي
        - label: التصنيف (إيجابي، محايد، سلبي)
        
        يمكنك أيضًا تحميل ملف CSV منفصل للتقييم (اختياري).
        """)
        
        # Upload training data
        train_file = st.file_uploader(
            "ملف التدريب (CSV):",
            type=['csv'],
            key="train_file"
        )
        
        # Upload evaluation data (optional)
        eval_file = st.file_uploader(
            "ملف التقييم (CSV، اختياري):",
            type=['csv'],
            key="eval_file"
        )
        
        if train_file:
            try:
                train_df = pd.read_csv(train_file)
                if not all(col in train_df.columns for col in ['text', 'label']):
                    st.error("⚠️ يجب أن يحتوي ملف التدريب على عمودي 'text' و 'label'")
                    return
                
                st.success(f"✅ تم تحميل ملف التدريب بنجاح. عدد النصوص: {len(train_df)}")
                
                # Show sample of training data
                st.markdown("### عينة من بيانات التدريب")
                st.dataframe(train_df.head())
                
                # Load evaluation data if provided
                eval_df = None
                if eval_file:
                    eval_df = pd.read_csv(eval_file)
                    if not all(col in eval_df.columns for col in ['text', 'label']):
                        st.error("⚠️ يجب أن يحتوي ملف التقييم على عمودي 'text' و 'label'")
                        return
                    st.success(f"✅ تم تحميل ملف التقييم بنجاح. عدد النصوص: {len(eval_df)}")
                
                # Training button
                if st.button("🚀 بدء التدريب", type="primary"):
                    with st.spinner("جاري تدريب النموذج..."):
                        # Evaluate current model first
                        if eval_df is not None:
                            st.markdown("### تقييم النموذج الحالي")
                            current_metrics = trainer.evaluate_model(current_version, eval_df)
                            st.metric("دقة النموذج الحالي", f"{current_metrics['eval_accuracy']:.2%}")
                        
                        # Train new model
                        version_info = trainer.train_sentiment_model(train_df, eval_df)
                        
                        # Show results
                        st.success(f"✅ تم تدريب النموذج بنجاح! الإصدار الجديد: {version_info['version']}")
                        
                        if eval_df is not None:
                            st.metric(
                                "تحسن الدقة",
                                f"{version_info['metrics']['eval_accuracy']:.2%}",
                                f"{version_info['metrics']['eval_accuracy'] - current_metrics['eval_accuracy']:.2%}"
                            )
                
                # Show version history
                st.markdown("### سجل إصدارات النموذج")
                history = trainer.get_version_history()
                
                if history:
                    history_df = pd.DataFrame(history)
                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                    history_df['accuracy'] = history_df['metrics'].apply(
                        lambda x: x.get('eval_accuracy', x.get('accuracy', 0))
                    )
                    
                    # Plot accuracy improvement
                    st.line_chart(history_df.set_index('version')['accuracy'])
                    
                    # Show detailed history
                    st.dataframe(
                        history_df[[
                            'version', 'timestamp', 'accuracy',
                            'train_samples', 'eval_samples'
                        ]],
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"❌ حدث خطأ: {str(e)}")
                logger.error(f"Training error: {e}")

if __name__ == "__main__":
    main()