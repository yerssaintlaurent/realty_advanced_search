import streamlit as st
import requests
import os
from PIL import Image
from pathlib import Path
import numpy as np
import base64
from transformers import pipeline, T5ForConditionalGeneration, T5TokenizerFast
import json
import re
from rapidfuzz import process, fuzz
from transformers import CLIPProcessor, CLIPModel
import torch

PHOTO_BASE_DIR = Path("photo")
DEFAULT_IMAGE = Image.new("RGB", (800, 600), color="#f0f0f0")
MODEL_NAME_EN = "ersace/bert_realestate_english"
MODEL_NAME_RU = "ersace/bert_realestate_rus"
MAX_LENGTH = 128

tokenizer_en = AutoTokenizer.from_pretrained(MODEL_NAME_EN)
model_en = AutoModelForTokenClassification.from_pretrained(MODEL_NAME_EN)
ner_pipeline = pipeline("ner", model=model_en, tokenizer=tokenizer_en)

def clean_text(text):
    return re.sub(r'\s?##', '', text)

processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

REFERENCE_FILTERS_EN = {
    "town": ["Almaty", "Astana", "Shymkent", "Aktau", "Atyrau", "Ust-Kamenogorsk", "Petropavl", "Karaganda", "Aktobe", "Oral", "Kostanay", "Pavlodar", "Taraz", "Kyzylorda", "Semey", "Kokshetau", "Temirtau", "Uralsk"],
    ...
}

REFERENCE_FILTERS_RU = {
    "town": ["Алматы", "Астана", "Актау", "Атырау", "Усть-каменогорск", "Петропавловск", "Караганда", "Шымкент", "Актобе", "Уральск", "Костанай", "Павлодар", "Тараз", "Кызылорда", "Семей", "Кокшетау", "Темиртау"],
    ...
}

UI_TRANSLATIONS = {
    'en': {
        'search_placeholder': '🔍 Search properties...',
        'filters_button': '⚙️ Filters',
        'max_price': 'Max Price (₸)',
        'max_area': 'Max Area (m²)',
        ...
    },
    'ru': {
        'search_placeholder': '🔍 Поиск недвижимости...',
        'filters_button': '⚙️ Фильтры',
        'max_price': 'Макс. цена (₸)',
        ...
    }
}

def extract_rooms_en(text):
    match = re.search(r'(?i)\b(\d+)\s*-?\s*(?:bedrooms?|rooms?)\b', text)
    if not match:
        match = re.search(r'^\D*(\d+)', text)
    return int(match.group(1)) if match else None

def extract_area_en(text):
    match = re.search(r'(\d+\.?\d*)\s*(?:sq\.? ?m|square meters?)', text, re.IGNORECASE)
    if not match:
        match = re.search(r'^(\d+\.?\d*)\b', text)
    return float(match.group(1)) if match else None

def extract_price_en(text):
    try:
        cleaned = re.sub(r'[^\d.,]', '', text.replace(',', '.'))
        match = re.search(r'(\d+\.?\d*?)(?:\.\d+)?$', cleaned)
        if match:
            price = int(float(match.group(1)))
            print(f"Extracted price: {price} from text: {text}")
            return price
        return None
    except (ValueError, TypeError) as e:
        print(f"Error extracting price from {text}: {str(e)}")
        return None

def normalize_text_en(text):
    return text.lower().strip()

def fuzzy_match_en(query, choices, threshold=80):
    norm_query = normalize_text_en(query)
    norm_choices = {normalize_text_en(c): c for c in choices}
    result = process.extractOne(norm_query, list(norm_choices.keys()), scorer=fuzz.WRatio)
    if result and result[1] >= threshold:
        return norm_choices[result[0]]
    return None

def map_filters_en(parsed_data):
    mapped = {}

    field_mapping = {
        'rooms': ('rooms', None, extract_rooms_en),
        'area': ('area', None, extract_area_en),
        ...
    }

    for source_field, (target_field, threshold, extractor) in field_mapping.items():
        if value := parsed_data.get(source_field):
            if extractor:
                processed_value = extractor(str(value))
                if processed_value is not None:
                    if source_field == 'rooms':
                        mapped[target_field] = [processed_value]
                    else:
                        mapped[target_field] = float(processed_value)
            else:
                if isinstance(value, list):
                    values = value
                else:
                    values = [value]
                    
                matched_values = []
                for val in values:
                    if match := fuzzy_match_en(str(val), REFERENCE_FILTERS_EN.get(source_field, []), threshold):
                        if match not in matched_values:
                            matched_values.append(match)
                            
                if matched_values:
                    mapped[target_field] = matched_values if len(matched_values) > 1 else matched_values[0]

    return mapped

def process_en_query(query: str):
    try:
        ner_results = ner_pipeline(query)
        parsed_data = {}
        current_entity = []
        current_label = None
        
        for token in ner_results:
            label = token['entity']
            value = token['word']

            if label.startswith('B-') or  label.startswith('I-'):
                label = label[2:]
            
            if label != current_label:
                if current_entity and current_label:
                    entity_text = ' '.join(current_entity)
                    entity_text = clean_text(entity_text)
                    if current_label in parsed_data:
                        if isinstance(parsed_data[current_label], list):
                            parsed_data[current_label].append(entity_text)
                        else:
                            parsed_data[current_label] = [parsed_data[current_label], entity_text]
                    else:
                        parsed_data[current_label] = entity_text
                current_entity = [value]
                current_label = label
            else:
                current_entity.append(value)
        
        if current_entity and current_label:
            entity_text = ' '.join(current_entity)
            entity_text = clean_text(entity_text)
            if current_label in parsed_data:
                if isinstance(parsed_data[current_label], list):
                    parsed_data[current_label].append(entity_text)
                else:
                    parsed_data[current_label] = [parsed_data[current_label], entity_text]
            else:
                parsed_data[current_label] = entity_text

        mapped_result = map_filters_en(parsed_data)
        
        if mapped_result:
            st.session_state.filters = {}

            filter_mapping = {
                'realty_types': 'type',
                'town': 'city',
                ...
            }
            
            for source_field, target_field in filter_mapping.items():
                if source_field in mapped_result:
                    value = mapped_result[source_field]
                    if source_field == 'price':
                        st.session_state.filters[target_field] = float(value)
                    elif source_field == 'rooms':
                        st.session_state.filters[target_field] = value if isinstance(value, list) else [value]
                    else:
                        st.session_state.filters[target_field] = value if isinstance(value, list) else [value]
        
        return parsed_data, mapped_result
        
    except Exception as e:
        st.error(f"Ошибка обработки запроса: {str(e)}")
        return {}, {}

def process_ru_query(query: str):
    try:
        tokenizer_ru = AutoTokenizer.from_pretrained(MODEL_NAME_RU)
        model_ru = AutoModelForTokenClassification.from_pretrained(MODEL_NAME_RU)
        ner_pipeline_ru = pipeline("ner", model=model_ru, tokenizer=tokenizer_ru)
        
        ner_results = ner_pipeline_ru(query)

        parsed_data = {}
        current_entity = []
        current_label = None
        
        for token in ner_results:
            label = token['entity']
            value = token['word']
            
            if label.startswith('B-') or label.startswith('I-'):
                label = label[2:]
            
            if label != current_label:
                if current_entity and current_label:
                    entity_text = ' '.join(current_entity)
                    entity_text = clean_text(entity_text)
                    if current_label in parsed_data:
                        if isinstance(parsed_data[current_label], list):
                            parsed_data[current_label].append(entity_text)
                        else:
                            parsed_data[current_label] = [parsed_data[current_label], entity_text]
                    else:
                        parsed_data[current_label] = entity_text
                current_entity = [value]
                current_label = label
            else:
                current_entity.append(value)

        if current_entity and current_label:
            entity_text = ' '.join(current_entity)
            entity_text = clean_text(entity_text)
            if current_label in parsed_data:
                if isinstance(parsed_data[current_label], list):
                    parsed_data[current_label].append(entity_text)
                else:
                    parsed_data[current_label] = [parsed_data[current_label], entity_text]
            else:
                parsed_data[current_label] = entity_text

        mapped_result = map_filters_ru(parsed_data)
        
        return parsed_data, mapped_result
        
    except Exception as e:
        st.error(f"Ошибка обработки запроса: {str(e)}")
        return {}, {}

def map_filters_ru(parsed_data):
    mapped = {}

    field_mapping = {
        'rooms': ('rooms', None, extract_rooms_ru),
        ...
    }

    for source_field, (target_field, threshold, extractor) in field_mapping.items():
        if value := parsed_data.get(source_field):
            if extractor:
                processed_value = extractor(str(value))
                if processed_value is not None:
                    if source_field == 'rooms':
                        mapped[target_field] = [processed_value]
                    else:
                        mapped[target_field] = float(processed_value)
            else:
                if isinstance(value, list):
                    values = value
                else:
                    values = [value]
                    
                matched_values = []
                for val in values:
                    if match := fuzzy_match_ru(str(val), REFERENCE_FILTERS_RU.get(source_field, []), threshold, source_field):
                        if match not in matched_values:
                            matched_values.append(match)
                            
                if matched_values:
                    mapped[target_field] = matched_values if len(matched_values) > 1 else matched_values[0]

    return mapped

def extract_rooms_ru(text):
    text_to_num = {
        ...
    }
    
    match = re.search(r'(?i)\b(\d+)\s*-?\s*(?:комнатная|комнат(?:ы|а)?|комн)\b', text)
    if match:
        return int(match.group(1))

    text = text.lower()
    for word, num in text_to_num.items():
        if any(pattern in text for pattern in [
            ...
        ]):
            return num

    match = re.search(r'^\D*(\d+)', text)
    return int(match.group(1)) if match else None

def extract_area_ru(text):
    match = re.search(r'(\d+\.?\d*)\s*(?:кв\.? ?м|квадратных метров?)', text, re.IGNORECASE)
    if not match:
        match = re.search(r'^(\d+\.?\d*)\b', text)
    return float(match.group(1)) if match else None

def extract_price_ru(text):
    ...
    except (ValueError, TypeError) as e:
        print(f"Error extracting price from {text}: {str(e)}")
        return None

def load_listing_images(listing_id):
    ...
    
    return images if images else [DEFAULT_IMAGE]

import random
REFERENCE_FILTERS_EN = {
    "town": ["Almaty", "Astana", "Shymkent", "Aktau", "Atyrau", "Ust-Kamenogorsk", 
             "Petropavl", "Karaganda", "Aktobe", "Oral", "Kostanay", "Pavlodar", 
             "Taraz", "Kyzylorda", "Semey", "Kokshetau", "Temirtau", "Uralsk"],
    
   ...
}

def init_state():
    if 'listings' not in st.session_state:
        st.session_state.listings = [
            {
                'id': i,
                'title': f"Property #{i}",
                'price': 30000 + i*5000,
                ...
            } 
            for i in range(1, 31)
        ]
    
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""

    if 'filters' not in st.session_state:
        st.session_state.filters = {}

def get_relevant_images(query: str, listings: list):
    try:
        parsed_data, mapped_result = process_en_query(query) if st.session_state.language == 'en' else process_ru_query(query)
        
        if query_parts := build_query_parts(parsed_data, mapped_result):
            text_query = ", ".join(query_parts)
            all_images = []
            image_indices = []
            listing_indices = []

            for listing_idx, listing in enumerate(listings):
                if listing['images']:
                    for img_idx, img in enumerate(listing['images']):
                        all_images.append(img)
                        image_indices.append(img_idx)
                        listing_indices.append(listing_idx)
            
            if all_images:
                inputs = processor(
                    text=[text_query],
                    images=all_images,
                    return_tensors="pt",
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    scores = logits_per_image.squeeze().tolist()

                listing_scores = {}
                for score, img_idx, listing_idx in zip(scores, image_indices, listing_indices):
                    if listing_idx not in listing_scores:
                        listing_scores[listing_idx] = []
                    listing_scores[listing_idx].append((score, img_idx))

                best_images = {}
                for listing_idx, scores_and_indices in listing_scores.items():
                    best_score, best_img_idx = max(scores_and_indices, key=lambda x: x[0])
                    best_images[listing_idx] = (best_score, best_img_idx)

                sorted_listings = []
                for listing_idx, (score, img_idx) in sorted(best_images.items(), key=lambda x: x[1][0], reverse=True):
                    listing = listings[listing_idx].copy()
                    st.session_state[f"img_idx_{listing['id']}"] = img_idx
                    sorted_listings.append(listing)
                
                return sorted_listings
        
        return listings
        
    except Exception as e:
        st.error(f"Ошибка при обработке изображений: {str(e)}")
        return listings

def build_query_parts(parsed_data, mapped_result):
    query_parts = []
    
    if 'realty_types' in mapped_result:
        query_parts.append(str(mapped_result['realty_types']))
    
    
    return query_parts

def main():
    CITY_MAP_RU = {
        "Almaty": "Алматы",
        ...
    }

    PROPERTY_TYPE_MAP_RU = {
        "apartment": "квартира",
        ...
    }
    st.set_page_config(layout="wide", page_title="Real Estate Search")

    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col3:
        if st.button("🌐 EN/RU"):
            st.session_state.language = 'ru' if st.session_state.language == 'en' else 'en'
            st.rerun()
    
    st.markdown("""
    <style>
        .listing-card {
            border-radius: 16px;
            padding: 16px;
            margin: 12px 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            transition: transform 0.2s;
            background: white;
            width: 100%;
            min-height: 500px;
        }
        .price-text { color: #ff385c; font-weight: 700; }
        .image-container { 
            position: relative; 
            margin-bottom: 12px;
            height: 240px;
            overflow: hidden;
        }
        .nav-button { 
            position: absolute; 
            top: 50%; 
            transform: translateY(-50%);
            background: rgba(255,255,255,0.7);
            color: #333;
            border: none;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            cursor: pointer;
            font-size: 18px;
            opacity: 0.8;
            transition: all 0.2s;
            z-index: 2;
        }
        .nav-button:hover {
            opacity: 1;
            background: rgba(255,255,255,0.9);
        }
        .prev-btn { left: 10px; }
        .next-btn { right: 10px; }
        img { 
            border-radius: 12px; 
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        .stTextInput > div > div > input {
            color: #4F4F4F;
            height: 38px !important;
            padding: 5px 10px !important;
        }
        .stTextInput > div > div > input::placeholder {
            color: #A9A9A9;
        }
        [data-testid="column"] {
            display: flex;
            align-items: center;
        }
        .stButton > button {
            height: 38px;
        }
    </style>
    """, unsafe_allow_html=True)

    debug_output = st.empty()

    with st.container():
        col1, col2 = st.columns([4, 1])
        current_lang = st.session_state.language
        translations = UI_TRANSLATIONS[current_lang]
    
        with col1:
            search_query = st.text_input(
                label="",
                placeholder=translations['search_placeholder'],
                value=st.session_state.get('last_query', ''),
                key="search_input"
            )
            if search_query and search_query != st.session_state.last_query:
                current_lang = st.session_state.language
                if current_lang == 'en':
                    parsed_data, mapped_result = process_en_query(search_query)
                else:
                    parsed_data, mapped_result = process_ru_query(search_query)
                
                # st.write("Parsed data:", parsed_data)
                # st.write("Mapped result:", mapped_result)

                if mapped_result:
                    st.session_state.filters = {}

                    for key, value in mapped_result.items():
                        st.session_state.filters[key] = value
                
                st.session_state.last_query = search_query
                st.rerun()

        with col2:
            st.markdown("<div style='margin-top: 28px;'>", unsafe_allow_html=True)
            if st.button("⚙️ Filters", use_container_width=True):
                st.session_state.show_filters = not st.session_state.get('show_filters', False)
            st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get('show_filters', False):
        with st.sidebar.expander("⚙️ FILTERS", expanded=True):
            current_lang = st.session_state.language
            translations = UI_TRANSLATIONS[current_lang]
            filters_data = REFERENCE_FILTERS_RU if current_lang == 'ru' else REFERENCE_FILTERS_EN

            current_price = st.session_state.filters.get('price_max', 100000)
            if isinstance(current_price, float):
                current_price = int(current_price)
            
            st.session_state.filters['price_max'] = st.slider(
                translations['max_price'],
                min_value=10000,
                max_value=100000,
                value=current_price,
                step=1000
            )
            
            current_area = st.session_state.filters.get('area', 500)
            if isinstance(current_area, float):
                current_area = int(current_area)
            
            st.session_state.filters['area'] = st.slider(
                translations['max_area'],
                min_value=20,
                max_value=500,
                value=current_area,
                step=1
            )

            rooms_options = list(range(1, 13))
            rooms_default = st.session_state.filters.get('rooms', [])
            rooms_default = [r for r in rooms_default if r in rooms_options]

            rooms = st.multiselect(
                translations['rooms'],
                options=rooms_options,
                default=rooms_default
            )
            st.session_state.filters['rooms'] = rooms
            
            st.session_state.filters['rooms'] = rooms

            ...

    filtered_listings = st.session_state.listings
    if st.session_state.get('filters'):
        filters = st.session_state.filters
        current_lang = st.session_state.language
        filtered_listings = [
            listing for listing in filtered_listings
            if (not filters.get('rooms') or any(r == listing['rooms'] for r in filters['rooms']))
            ...
        ]

    if search_query:
        filtered_listings = get_relevant_images(search_query, filtered_listings)

    CITY_MAP_RU = {
        "Almaty": "Алматы",
        ...
    }

    PROPERTY_TYPE_MAP_RU = {
        "apartment": "квартира",
        ...
    }

    cols = st.columns(3)
    current_lang = st.session_state.language

    for idx, listing in enumerate(filtered_listings):
        with cols[idx % 3]:
            with st.container():
                img_html = ""
                img_idx = st.session_state.get(f"img_idx_{listing['id']}", 0)
                if listing['images']:
                    img_base64 = get_image_base64(listing['images'][img_idx])
                    img_html = f'<img src="data:image/jpeg;base64,{img_base64}" width="100%">'

                if current_lang == 'ru':
                    title_text = f"Объект #{listing['id']}"
                    city_display = CITY_MAP_RU.get(listing['city'], listing['city'])
                    type_display = PROPERTY_TYPE_MAP_RU.get(listing['type'], listing['type'])
                    area_unit = "м²"
                    rooms_unit = "комнат"
                else:
                    title_text = f"Property #{listing['id']}"
                    city_display = listing['city']
                    type_display = listing['type']
                    area_unit = "m²"
                    rooms_unit = "rooms"
                
                st.markdown(f"""
                <div class="listing-card">
                    <div class="image-container">
                        {img_html}
                    </div>
                    <h3>{title_text}</h3>
                    <p class="price-text">{listing['price']} ₸</p>
                    <p>📍 {city_display} • 🏠 {type_display}</p>
                    <p>📏 {listing['area']} {area_unit} • 🛏 {listing['rooms']} {rooms_unit}</p>
                </div>
                """, unsafe_allow_html=True)

                if len(listing['images']) > 1:
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("←", key=f"prev_{listing['id']}"):
                            st.session_state[f"img_idx_{listing['id']}"] = max(0, img_idx - 1)
                    with c2:
                        if st.button("→", key=f"next_{listing['id']}"):
                            st.session_state[f"img_idx_{listing['id']}"] = min(
                                len(listing['images']) - 1, img_idx + 1)

@st.cache_data
def _get_image_bytes(image_bytes):
    return base64.b64encode(image_bytes).decode()

def get_image_base64(image):
    try:
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        return _get_image_bytes(img_bytes)
    except Exception as e:
        st.error(f"Ошибка конвертации изображения: {str(e)}")
        return ""

def normalize_text(text):
    return text.lower().strip()

def fuzzy_match(query, choices, threshold=60):
    try:
        result = process.extractOne(
            normalize_text(query),
            choices,
            scorer=fuzz.WRatio,
            score_cutoff=threshold
        )
        if result:
            return choices[choices.index(result[0])]
        return None
    except Exception as e:
        st.error(f"Ошибка при сопоставлении: {str(e)}")
        return None

def normalize_text_ru(text):
    text = text.lower().strip()

    action_mappings = {
        ....
    }

    for old, new in action_mappings.items():
        if old in text:
            text = text.replace(old, new)
    
    return text.strip()

def fuzzy_match_ru(query, choices, threshold=60, category=None):
    if not choices:
        return None
    
    norm_query = normalize_text_ru(query)

    if category == 'action_types':
        action_mappings = {
            ...
        }
        for base_action, synonyms in action_mappings.items():
            if norm_query in [normalize_text_ru(s) for s in synonyms]:
                return base_action

    norm_choices = {normalize_text_ru(c): c for c in choices}
    result = process.extractOne(norm_query, list(norm_choices.keys()), scorer=fuzz.WRatio)
    if result and result[1] >= threshold:
        return norm_choices[result[0]]
    return None

port = int(os.environ.get("PORT", 8501))  # Используем порт, который даёт Render

if __name__ == "__main__":
    st.run(port=port, host="0.0.0.0")
