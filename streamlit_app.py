import streamlit as st
import requests
import os
from PIL import Image
from pathlib import Path
import numpy as np
import base64
from transformers import pipeline, T5ForConditionalGeneration, T5TokenizerFast
import torch
import json
import re
from rapidfuzz import process, fuzz
from transformers import CLIPProcessor, CLIPModel
import torch

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import re

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
    "district": ["Medeu", "downtown", "outskirts", "city center", "Bostandyk", "Alatau", "Almaly", "Auezov", "Zhetysu", "Nauryzbay", "Turksib"],
    "realty_types": ["studio flat", "villa", "apartment", "flat", "room", "house", "cottage", "land plot", "garage", "hostel", "hotel", "motel", "guest house", "apart-hotel"],
    "action_types": ["rent", "short-term rent", "long-term rent", "buy", "sell"],
    "is_by_homeowner": ["owner", "realtor"],
    "photo": ["with photo", "without photo"],
    "comfort": ["pets allowed", "free wifi", "soundproofing", "separate bedroom", "charging station", "robot vacuum", "home theater", "projector", "mountain view", "smart lock", "smart TV", "high-speed internet"],
    'entertainment': ['swimming pool', 'mini bar', 'jacuzzi', 'LED lighting', 'game console', 'board games'],
    "climate_control": ["air conditioning", "fan", "heater"],
    'apart_features': ['balcony', 'unfurnished', 'cosmetic repairs', 'furnished'],
    "parc": ["free parking", "underground parking", "paid parking"],
    "location_features": ["quiet area", "supermarket", "downtown", "yard view", "city view", "park view", "waterfront view", "skyscraper view", "river view", "sea view", "school", "park"],
    "inter_work": ["workspace"],
    "kitchen": ["kitchen", "gas stove", "dining set", "dining area", "electric stove", "drinking water", "refrigerator", "dishes", "sweets", "coffee machine", "microwave", "walk-in pantry"],
    "photo": ["with photo", "without photo"],
    'family': ['car-sharing', 'baby crib', 'family'],
    'apart_security': ['gated community', '24/7 security', 'guarded entrance', 'CCTV cameras', 'elevator', 'smart lock', 'video intercom', 'security'],
    'bedroom_bath': ['shower', 'laundry', 'hygiene products', 'iron', 'washing machine'],
    'nearby': [
            'public transport', 'restaurant', 'coffee shop', 'cafe',
            'metro station', 'bus stop', 'airport', 'hospital',
            'pharmacy', 'clinic', 'sports complex', 'cinema',
            'shopping mall', 'gym', 'spa center', 'car rental',
            'bicycle parking', 'playground', 'beauty salon',
            'store', 'sports ground'],
    
    'international_student': ['international student'],
    
    'expat_friendly': [
            'expat-friendly', 'english-speaking landlord',
            'international community', 'embassy', 'visa support',
            'foreigner registration assistance', 'international school',
            'business center', 'diplomatic district']
}

REFERENCE_FILTERS_RU = {
    "town": ["Алматы", "Астана", "Актау", "Атырау", "Усть-каменогорск", "Петропавловск", "Караганда", "Шымкент", "Актобе", "Уральск", "Костанай", "Павлодар", "Тараз", "Кызылорда", "Семей", "Кокшетау", "Темиртау"],
    "district": ["Медеуский", "центр", "окраина", "центр города", "Бостандыкский", "Алатауский", "Алмалинский", "Ауэзовский", "Жетысуский", "Наурызбайский", "Турксибский"],
    "realty_types": ["вилла", "коливинг", "помещение", "квартира", "комната", "дом", "участок", "гараж", "жилье", "недвижимость", "хостел", "гостиница", "гостиничный номер", "гостевой дом", "апарт-отель"],
    "action_types": ["продажа", "аренда", "купить"],
    "is_by_homeowner": ["собственник", "хозяин", "риелтор", "агенство", "без посредников", "риэлтор"],
    "period": ["месяц", "долгосрок", "долгосрочный", "день", "ночь", "вечер", "два дня", "один день", "три дня", "четыре дня", "пять дней", "дней", "шесть дней", "семь дней", "неделя", "1 день", "2 дня", "3 дня", "4 дня", "5 дней", "полгода", "пол года", "год", "посуточно", "длительный", "на долго"],
    "photo": ["с фото", "без фото"],
    "comfort": ["с животными", "хорошая шумоизоляция", "отдельная спальня", "зарядная станция", "робот пылесос", "домашний кинотеатр", "проектор", "кондиционер", "вид на горы", "в горах", "торговый центр", "бесплатный вайфай", "бесплатный wifi", "бесплатный Wi Fi", "электронные замки", "smart tv"],
    "entertainment": ["бассейн", "джакузи", "мини-бар", "мини бар", "мини-баром", "smart tv", "led-освещение", "игровая приставка", "настольные игры"],
    "family": ["каршеринг", "детская кроватка"],
    "apart_features": ["без мебели", "косметический ремонт", "евроремонт"],
    "apart_security": ["лифт", "электронные замки", "видеодомофон", "домофон", "охрана"],
    "inter_work": ["рабочая зона"],
    "kitchen": ["газовая плита", "столовый прибор", "электроплита", "питьевая вода", "холодильник", "посуда", "конфеты и сладости", "кофеварка", "кофемашина", "микроволновка"],
    "location_features": ["тихий район", "супермаркет", "центр города", "вид во двор", "вид на город", "вид на парк", "вид на набережную", "вид на высотные здания", "вид на море", "вид на реку", "вид на водоем", "учебное заведение", "активный район", "ресторан", "парк", "РОВД", "кофейня", "компьютерный клуб", "школа"],
    "parc": ["бесплатная парковка", "подземный паркинг", "платная парковка"],
    "climate_control": ["вентилятор", "кондиционер", "обогреватель"],
    "bedroom_bath": ["средства личной гигиены", "утюг", "стиральная машина"],
    "nearby": ["метро", "детская площадка", "салон красоты", "аптека", "магазин", "автобусная остановка", "спортивная площадка"],
    "international_student": ["для студентов", "для иностранных студентов"],
    "expat_friendly": ["для экспатов", "англоговорящий владелец", "международное сообщество", "рядом с посольством", "визовая поддержка", "помощь с регистрацией", "международная школа", "бизнес центр", "дипломатический район"]
}

UI_TRANSLATIONS = {
    'en': {
        'search_placeholder': '🔍 Search properties...',
        'filters_button': '⚙️ Filters',
        'max_price': 'Max Price (₸)',
        'max_area': 'Max Area (m²)',
        'rooms': 'Number of Rooms',
        'listing_type': 'Property Type',
        'action': 'Action',
        'city': 'City',
        'district': 'District',
        'comfort': 'Comfort',
        'entertainment': 'Entertainment',
        'climate_control': 'Climate Control',
        'apart_features': 'Apartment Features',
        'parking': 'Parking',
        'location': 'Location',
        'workspace': 'Workspace',
        'kitchen': 'Kitchen',
        'family': 'Family',
        'security': 'Security',
        'bathroom': 'Bathroom',
        'nearby': 'Nearby',
        'international': 'International Student',
        'expat': 'Expat Friendly',
        'no_interior': 'No interior description found in the query. Using standard filtering.'
    },
    'ru': {
        'search_placeholder': '🔍 Поиск недвижимости...',
        'filters_button': '⚙️ Фильтры',
        'max_price': 'Макс. цена (₸)',
        'max_area': 'Макс. площадь (м²)',
        'rooms': 'Количество комнат',
        'listing_type': 'Тип недвижимости',
        'action': 'Действие',
        'city': 'Город',
        'district': 'Район',
        'comfort': 'Удобства',
        'entertainment': 'Развлечения',
        'climate_control': 'Климат-контроль',
        'apart_features': 'Особенности квартиры',
        'parking': 'Парковка',
        'location': 'Расположение',
        'workspace': 'Рабочая зона',
        'kitchen': 'Кухня',
        'family': 'Семья',
        'security': 'Безопасность',
        'bathroom': 'Ванная',
        'nearby': 'Рядом',
        'expat': 'Для экспатов',
        'no_interior': 'Описание интерьера не найдено в запросе. Используется стандартная фильтрация.'
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
        'price': ('price_max', None, extract_price_en),
        'town': ('city', 80, None),
        'district': ('district', 75, None),
        'realty_types': ('type', 70, None),
        'action_types': ('action', 65, None),
        'comfort': ('comfort', 65, None),
        'entertainment': ('entertainment', 65, None),
        'apart_features': ('apart_features', 70, None),
        'apart_security': ('security', 70, None),
        'inter_work': ('inter_work', 75, None),
        'kitchen': ('kitchen', 70, None),
        'location_features': ('location_features', 65, None),
        'parc': ('pac', 75, None),
        'climate_control': ('climate_control', 70, None),
        'bedroom_bath': ('bedroom_bath', 65, None),
        'nearby': ('nearby', 65, None),
        'family': ('family', 70, None)
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
                'action_types': 'action',
                'rooms': 'rooms',
                'price': 'price_max',
                'area': 'area',
                'comfort': 'comfort',
                'entertainment': 'entertainment',
                'climate_control': 'climate_control',
                'apart_features': 'apart_features',
                'parc': 'parc',
                'location_features': 'location_features',
                'inter_work': 'inter_work',
                'kitchen': 'kitchen',
                'family': 'family',
                'apart_security': 'apart_security',
                'bedroom_bath': 'bedroom_bath',
                'nearby': 'nearby',
                'international_student': 'international_student',
                'expat_friendly': 'expat_friendly',
                'district': 'district'
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
        'area': ('area', None, extract_area_ru),
        'price': ('price_max', None, extract_price_ru),
        'town': ('city', 80, None),
        'district': ('district', 75, None),
        'realty_types': ('type', 70, None),
        'action_types': ('action', 65, None),
        'comfort': ('comfort', 65, None),
        'entertainment': ('entertainment', 65, None),
        'apart_features': ('apart_features', 70, None),
        'apart_security': ('apart_security', 70, None),
        'inter_work': ('inter_work', 75, None),
        'kitchen': ('kitchen', 70, None),
        'location_features': ('location_features', 65, None),
        'parc': ('parc', 75, None),
        'climate_control': ('climate_control', 70, None),
        'bedroom_bath': ('bedroom_bath', 65, None),
        'nearby': ('nearby', 65, None),
        'family': ('family', 70, None),
        'international_student': ('international_student', 75, None),
        'expat_friendly': ('expat_firendly', 65, None)
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
        'одн': 1, 'один': 1, 'одна': 1, 'однушк': 1, 'однушек': 1,
        'дв': 2, 'два': 2, 'двух': 2, 'двушк': 2, 'двушек': 2,
        'тр': 3, 'три': 3, 'трёх': 3, 'трех': 3, 'трешк': 3, 'трёшк': 3,
        'четыр': 4, 'четырёх': 4, 'четырех': 4,
        'пят': 5, 'пяти': 5,
        'шест': 6, 'шести': 6,
        'сем': 7, 'семи': 7,
        'восем': 8, 'восьми': 8,
        'девят': 9, 'девяти': 9,
        'десят': 10, 'десяти': 10
    }
    
    match = re.search(r'(?i)\b(\d+)\s*-?\s*(?:комнатная|комнат(?:ы|а)?|комн)\b', text)
    if match:
        return int(match.group(1))

    text = text.lower()
    for word, num in text_to_num.items():
        if any(pattern in text for pattern in [
            f"{word}комнатную",
            f"{word}комнатная",
            f"{word} комнатную",
            f"{word} комнатная",
            f"{word}ушка",
            f"{word}ушку",
            f"{word}ёшка",
            f"{word}ешка",
            f"{word}ёшку",
            f"{word}ешку"
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
    try:
        if 'тыс' in text.lower():
            cleaned = re.sub(r'[^\d.,]', '', text.replace(',', '.'))
            match = re.search(r'(\d+\.?\d*?)(?:\.\d+)?$', cleaned)
            if match:
                price = int(float(match.group(1)) * 1000)
                print(f"Extracted price: {price} from text with thousands: {text}")
                return price
        
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

def load_listing_images(listing_id):
    image_dir = PHOTO_BASE_DIR / str(listing_id)
    images = []
    
    if image_dir.exists() and image_dir.is_dir():
        for img_path in sorted(image_dir.glob("*.jpg")):
            try:
                images.append(Image.open(img_path))
            except Exception as e:
                st.error(f"Error loading image {img_path}: {str(e)}")
    
    return images if images else [DEFAULT_IMAGE]

import random
REFERENCE_FILTERS_EN = {
    "town": ["Almaty", "Astana", "Shymkent", "Aktau", "Atyrau", "Ust-Kamenogorsk", 
             "Petropavl", "Karaganda", "Aktobe", "Oral", "Kostanay", "Pavlodar", 
             "Taraz", "Kyzylorda", "Semey", "Kokshetau", "Temirtau", "Uralsk"],
    
    "district": ["Medeu", "downtown", "outskirts", "city center", "Bostandyk", 
                 "Alatau", "Almaly", "Auezov", "Zhetysu", "Nauryzbay", "Turksib"],
    
    "realty_types": ["studio flat", "villa", "apartment", "flat", "room", 
                     "house", "cottage", "land plot", "garage", "hostel",
                     "hotel", "motel", "guest house", "apart-hotel"],
    
    "action_types": ["rent", "short-term rent", "long-term rent", "buy", "sell"],
    
    "comfort": ["pets allowed", "free wifi", "soundproofing", "separate bedroom", 
                "charging station", "robot vacuum", "home theater", "projector",
                "mountain view", "smart lock", "smart TV", "high-speed internet"],
    
    "entertainment": ["swimming pool", "mini bar", "jacuzzi",
                      "LED lighting", "game console", "board games"],
    
    "climate_control": ["air conditioning", "fan", "heater"],
    
    "apart_features": ["balcony", "unfurnished", "cosmetic repairs", "furnished"],
    
    "parc": ["free parking", "underground parking", "paid parking"],
    
    "location_features": ["quiet area", "supermarket", "downtown", "yard view", "city view", "park view", "waterfront view",
                          "skyscraper view", "river view", "sea view", "school", "park"],
    
    "inter_work": ["workspace"],
    
    "kitchen": ["kitchen", "gas stove", "dining set", "dining area", "electric stove", "drinking water", "refrigerator", "dishes", "sweets",
                "coffee machine", "microwave", "walk-in pantry"],
    
    "family": ["car-sharing", "baby crib", "family"],
    
    "apart_security": [
            "gated community", "24/7 security", "guarded entrance",
            "CCTV cameras", "elevator", "smart lock",
            "video intercom", "security"],
    
    "bedroom_bath": [
            "shower", "laundry", "hygiene products",
            "iron", "washing machine"],
    
    "nearby": [
            "public transport", "restaurant", "coffee shop", "cafe",
            "metro station", "bus stop", "airport", "hospital",
            "pharmacy", "clinic", "sports complex", "cinema",
            "shopping mall", "gym", "spa center", "car rental",
            "bicycle parking", "playground", "beauty salon",
            "store", "sports ground"],
    
    "international_student": ["international student"],
    
    "expat_friendly": [
            "expat-friendly", "english-speaking landlord",
            "international community", "embassy", "visa support",
            "foreigner registration assistance", "international school",
            "business center", "diplomatic district"]
}

def init_state():
    if 'listings' not in st.session_state:
        st.session_state.listings = [
            {
                'id': i,
                'title': f"Property #{i}",
                'price': 30000 + i*5000,
                'area': 50 + i*5,
                'rooms': i % 5 + 1,
                'city': random.choice(REFERENCE_FILTERS_EN.get("town", ["Almaty", "Astana"])),
                'type': random.choice(["apartment", "house"]),
                'action': random.choice(REFERENCE_FILTERS_EN.get("action_types", ["rent"])),
                'district': random.choice(REFERENCE_FILTERS_EN.get("district", ["downtown"])),
                'comfort': random.sample(REFERENCE_FILTERS_EN.get("comfort", []),
                                          k=min(2, len(REFERENCE_FILTERS_EN.get("comfort", [])))),
                'entertainment': random.sample(REFERENCE_FILTERS_EN.get("entertainment", []),
                                               k=min(2, len(REFERENCE_FILTERS_EN.get("entertainment", [])))),
                'climate_control': random.sample(REFERENCE_FILTERS_EN.get("climate_control", []),
                                                 k=min(1, len(REFERENCE_FILTERS_EN.get("climate_control", [])))),
                'apart_features': random.sample(REFERENCE_FILTERS_EN.get("apart_features", []),
                                                k=min(1, len(REFERENCE_FILTERS_EN.get("apart_features", [])))),
                'parc': random.sample(REFERENCE_FILTERS_EN.get("parc", []),
                                     k=min(1, len(REFERENCE_FILTERS_EN.get("parc", [])))),
                'location_features': random.sample(REFERENCE_FILTERS_EN.get("location_features", []),
                                                    k=min(1, len(REFERENCE_FILTERS_EN.get("location_features", [])))),
                'inter_work': random.sample(REFERENCE_FILTERS_EN.get("inter_work", []),
                                             k=min(1, len(REFERENCE_FILTERS_EN.get("inter_work", [])))),
                'kitchen': random.sample(REFERENCE_FILTERS_EN.get("kitchen", []),
                                          k=min(2, len(REFERENCE_FILTERS_EN.get("kitchen", [])))),
                'family': random.sample(REFERENCE_FILTERS_EN.get("family", []),
                                         k=min(1, len(REFERENCE_FILTERS_EN.get("family", [])))),
                'apart_security': random.sample(REFERENCE_FILTERS_EN.get("apart_security", []),
                                                 k=min(1, len(REFERENCE_FILTERS_EN.get("apart_security", [])))),
                'bedroom_bath': random.sample(REFERENCE_FILTERS_EN.get("bedroom_bath", []),
                                               k=min(1, len(REFERENCE_FILTERS_EN.get("bedroom_bath", [])))),
                'nearby': random.sample(REFERENCE_FILTERS_EN.get("nearby", []),
                                         k=min(2, len(REFERENCE_FILTERS_EN.get("nearby", [])))),
                'international_student': random.sample(REFERENCE_FILTERS_EN.get("international_student", []),
                                                        k=min(1, len(REFERENCE_FILTERS_EN.get("international_student", [])))),
                'expat_friendly': random.sample(REFERENCE_FILTERS_EN.get("expat_friendly", []),
                                                 k=min(1, len(REFERENCE_FILTERS_EN.get("expat_friendly", [])))),
                'images': load_listing_images(i)
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
    if 'town' in mapped_result:
        query_parts.append(f"in {str(mapped_result['town'])}")

    if interior_desc := parsed_data.get('interior_describing'):
        query_parts.append(str(interior_desc))
    
    return query_parts

def main():
    CITY_MAP_RU = {
        "Almaty": "Алматы",
        "Astana": "Астана",
        "Shymkent": "Шымкент",
        "Aktau": "Актау",
        "Atyrau": "Атырау",
        "Ust-Kamenogorsk": "Усть-каменогорск",
        "Petropavl": "Петропавловск",
        "Karaganda": "Караганда",
        "Aktobe": "Актобе",
        "Oral": "Уральск",
        "Kostanay": "Костанай",
        "Pavlodar": "Павлодар",
        "Taraz": "Тараз",
        "Kyzylorda": "Кызылорда",
        "Semey": "Семей",
        "Kokshetau": "Кокшетау",
        "Temirtau": "Темиртау"
    }

    PROPERTY_TYPE_MAP_RU = {
        "apartment": "квартира",
        "house": "дом",
        "room": "комната",
        "hostel": "хостел",
        "hotel": "гостиница",
        "villa": "вилла",
        "flat": "квартира",
        "cottage": "коттедж",
        "garage": "гараж",
        "apart-hotel": "апарт-отель"
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

            default_type = st.session_state.filters.get('type', [])
            if not isinstance(default_type, list):
                default_type = [default_type]

            listing_type = st.multiselect(
                translations['listing_type'],
                options=filters_data["realty_types"],
                default=default_type
            )
            st.session_state.filters['type'] = listing_type

            actions = st.multiselect(
                translations['action'],
                options=filters_data["action_types"],
                default=st.session_state.filters.get('action', [])
            )
            st.session_state.filters['action'] = actions

            cities = st.multiselect(
                translations['city'],
                options=filters_data["town"],
                default=st.session_state.filters.get('city', [])
            )
            st.session_state.filters['city'] = cities

            districts = st.multiselect(
                translations['district'],
                options=filters_data["district"],
                default=st.session_state.filters.get('district', [])
            )
            st.session_state.filters['district'] = districts

            comfort = st.multiselect(
                translations['comfort'],
                options=filters_data["comfort"],
                default=st.session_state.filters.get('comfort', [])
            )
            st.session_state.filters['comfort'] = comfort

            entertainment = st.multiselect(
                translations['entertainment'],
                options=filters_data["entertainment"],
                default=st.session_state.filters.get('entertainment', [])
            )
            st.session_state.filters['entertainment'] = entertainment

            climate = st.multiselect(
                translations['climate_control'],
                options=filters_data["climate_control"],
                default=st.session_state.filters.get('climate_control', [])
            )
            st.session_state.filters['climate_control'] = climate

            features = st.multiselect(
                translations['apart_features'],
                options=filters_data["apart_features"],
                default=st.session_state.filters.get('apart_features', [])
            )
            st.session_state.filters['apart_features'] = features

            parking = st.multiselect(
                translations['parking'],
                options=filters_data["parc"],
                default=st.session_state.filters.get('parc', [])
            )
            st.session_state.filters['parc'] = parking

            location = st.multiselect(
                translations['location'],
                options=filters_data["location_features"],
                default=st.session_state.filters.get('location_features', [])
            )
            st.session_state.filters['location_features'] = location

            workspace = st.multiselect(
                translations['workspace'],
                options=filters_data["inter_work"],
                default=st.session_state.filters.get('inter_work', [])
            )
            st.session_state.filters['inter_work'] = workspace

            kitchen = st.multiselect(
                translations['kitchen'],
                options=filters_data["kitchen"],
                default=st.session_state.filters.get('kitchen', [])
            )
            st.session_state.filters['kitchen'] = kitchen

            family = st.multiselect(
                translations['family'],
                options=filters_data["family"],
                default=st.session_state.filters.get('family', [])
            )
            st.session_state.filters['family'] = family

            security = st.multiselect(
                translations['security'],
                options=filters_data["apart_security"],
                default=st.session_state.filters.get('apart_security', [])
            )
            st.session_state.filters['apart_security'] = security

            bathroom = st.multiselect(
                translations['bathroom'],
                options=filters_data["bedroom_bath"],
                default=st.session_state.filters.get('bedroom_bath', [])
            )
            st.session_state.filters['bedroom_bath'] = bathroom

            nearby = st.multiselect(
                translations['nearby'],
                options=filters_data["nearby"],
                default=st.session_state.filters.get('nearby', [])
            )
            st.session_state.filters['nearby'] = nearby

    filtered_listings = st.session_state.listings
    if st.session_state.get('filters'):
        filters = st.session_state.filters
        current_lang = st.session_state.language
        filtered_listings = [
            listing for listing in filtered_listings
            if (not filters.get('rooms') or any(r == listing['rooms'] for r in filters['rooms']))
            and (not filters.get('city') or ((CITY_MAP_RU.get(listing['city'], listing['city'])) if current_lang == 'ru' else listing['city']) in filters['city'])
            and (not filters.get('action') or listing['action'] in filters['action'])
            and (not filters.get('price_max') or listing['price'] <= filters['price_max'])
            and (not filters.get('area') or listing['area'] <= filters['area'])
            and (not filters.get('type') or ((PROPERTY_TYPE_MAP_RU.get(listing['type'], listing['type'])) if current_lang == 'ru' else listing['type']) in filters['type'])
            and (not filters.get('district') or listing.get('district', "") in filters['district'])
            and (not filters.get('comfort') or any(c in listing.get('comfort', []) for c in filters['comfort']))
            and (not filters.get('entertainment') or any(e in listing.get('entertainment', []) for e in filters['entertainment']))
            and (not filters.get('climate_control') or any(cc in listing.get('climate_control', []) for cc in filters['climate_control']))
            and (not filters.get('apart_features') or any(af in listing.get('apart_features', []) for af in filters['apart_features']))
            and (not filters.get('parc') or any(p in listing.get('parc', []) for p in filters['parc']))
            and (not filters.get('location_features') or any(lf in listing.get('location_features', []) for lf in filters['location_features']))
            and (not filters.get('inter_work') or any(iw in listing.get('inter_work', []) for iw in filters['inter_work']))
            and (not filters.get('kitchen') or any(k in listing.get('kitchen', []) for k in filters['kitchen']))
            and (not filters.get('family') or any(f in listing.get('family', []) for f in filters['family']))
            and (not filters.get('apart_security') or any(as_ in listing.get('apart_security', []) for as_ in filters['apart_security']))
            and (not filters.get('bedroom_bath') or any(bb in listing.get('bedroom_bath', []) for bb in filters['bedroom_bath']))
            and (not filters.get('nearby') or any(n in listing.get('nearby', []) for n in filters['nearby']))
            and (not filters.get('international_student') or any(is_ in listing.get('international_student', []) for is_ in filters['international_student']))
            and (not filters.get('expat_friendly') or any(ef in listing.get('expat_friendly', []) for ef in filters['expat_friendly']))
        ]

    if search_query:
        filtered_listings = get_relevant_images(search_query, filtered_listings)

    CITY_MAP_RU = {
        "Almaty": "Алматы",
        "Astana": "Астана",
        "Shymkent": "Шымкент",
        "Aktau": "Актау",
        "Atyrau": "Атырау",
        "Ust-Kamenogorsk": "Усть-каменогорск",
        "Petropavl": "Петропавловск",
        "Karaganda": "Караганда",
        "Aktobe": "Актобе",
        "Oral": "Уральск",
        "Kostanay": "Костанай",
        "Pavlodar": "Павлодар",
        "Taraz": "Тараз",
        "Kyzylorda": "Кызылорда",
        "Semey": "Семей",
        "Kokshetau": "Кокшетау",
        "Temirtau": "Темиртау"
    }

    PROPERTY_TYPE_MAP_RU = {
        "apartment": "квартира",
        "house": "дом",
        "room": "комната",
        "hostel": "хостел",
        "hotel": "гостиница",
        "villa": "вилла",
        "flat": "квартира",
        "cottage": "коттедж",
        "garage": "гараж",
        "apart-hotel": "апарт-отель"
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
        'снять': 'аренда',
        'сниму': 'аренда',
        'арендовать': 'аренда',
        'арендую': 'аренда',
        'купить': 'продажа',
        'куплю': 'продажа',
        'продать': 'продажа',
        'продам': 'продажа'
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
            'аренда': ['снять', 'сниму', 'арендовать', 'арендую'],
            'продажа': ['купить', 'куплю', 'продать', 'продам']
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
