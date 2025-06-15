#import streamlit as st
#import numpy as np
#import cv2
#from PIL import Image
#import easyocr
#from spellchecker import SpellChecker
#
#st.set_page_config(page_title="OCR Intelligent FR avec EasyOCR", layout="centered")
#
## Charger EasyOCR une seule fois
#@st.cache_resource
#def load_easyocr_model():
#    return easyocr.Reader(['fr'], gpu=False)
#
#reader = load_easyocr_model()
#spell = SpellChecker(language='fr')
#
## Traitement complet de l'image (rotation, nettoyage, etc.)
#def preprocess_image(image_pil):
#    image = np.array(image_pil.convert("RGB"))
#    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
#    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#    enhanced = clahe.apply(denoised)
#
#    # Amélioration de la correction de rotation avec Canny et contours
#    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
#    edged = cv2.Canny(blurred, 50, 150)
#    coords = np.column_stack(np.where(edged > 0))
#    if coords.shape[0] > 0:
#        angle = cv2.minAreaRect(coords)[-1]
#        if angle < -45:
#            angle = -(90 + angle)
#        else:
#            angle = -angle
#        (h, w) = image.shape[:2]
#        center = (w // 2, h // 2)
#        M = cv2.getRotationMatrix2D(center, angle, 1.0)
#        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#
#    # Retourner l'image de 180° si le texte est à l'envers
#    image = cv2.rotate(image, cv2.ROTATE_180)
#    return image
#
## Interface utilisateur
#st.title("OCR Intelligent - EasyOCR (Français)")
#uploaded_file = st.file_uploader("Charge une image de document", type=["jpg", "jpeg", "png"])
#
#if uploaded_file:
#    original_image = Image.open(uploaded_file)
#    st.image(original_image, caption="Image originale", use_container_width=True)
#
#    processed_image = preprocess_image(original_image)
#
#    st.subheader("Zones détectées")
#    result_with_boxes = reader.readtext(processed_image)
#    image_with_boxes = processed_image.copy()
#
#    for (bbox, text, conf) in result_with_boxes:
#        (top_left, top_right, bottom_right, bottom_left) = bbox
#        top_left = tuple(map(int, top_left))
#        bottom_right = tuple(map(int, bottom_right))
#        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
#        cv2.putText(image_with_boxes, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
#
#    st.image(image_with_boxes, caption="Zones de texte détectées", use_container_width=True)
#
#    st.subheader("Texte extrait (ligne par ligne)")
#    line_texts = [text for (_, text, _) in result_with_boxes if text is not None]
#    raw_text = "\n".join(line_texts)
#
#    st.text_area("Texte brut OCR :", raw_text, height=200)
#
#    words = raw_text.split()
#    corrected_words = [spell.correction(word) if word and word.isalpha() else word for word in words]
#    corrected_text = " ".join([w for w in corrected_words if w])
#
#    st.subheader("Texte corrigé automatiquement")
#    st.text_area("Texte corrigé :", corrected_text, height=200)





####import streamlit as st
####import numpy as np
####import cv2
####import pandas as pd
####import json
####import os
####from PIL import Image
####import easyocr
####from spellchecker import SpellChecker
####
####st.set_page_config(page_title="OCR Intelligent FR avec EasyOCR", layout="centered")
####
####@st.cache_resource
####def load_easyocr_model():
####    return easyocr.Reader(['fr'], gpu=False)
####
####reader = load_easyocr_model()
####spell = SpellChecker(language='fr')
####DB_FILE = "corrections.json"
####
##### Charger ou initialiser base de corrections manuelles
####def load_corrections():
####    if os.path.exists(DB_FILE):
####        with open(DB_FILE, "r", encoding="utf-8") as f:
####            return json.load(f)
####    return {}
####
##### Sauvegarder corrections
####def save_corrections(corrections):
####    with open(DB_FILE, "w", encoding="utf-8") as f:
####        json.dump(corrections, f, ensure_ascii=False, indent=2)
####
####manual_corrections = load_corrections()
####
##### Prétraitement image
####def preprocess_image(image_pil):
####    image = np.array(image_pil.convert("RGB"))
####    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
####    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
####    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
####    enhanced = clahe.apply(denoised)
####
####    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
####    edged = cv2.Canny(blurred, 50, 150)
####    coords = np.column_stack(np.where(edged > 0))
####    if coords.shape[0] > 0:
####        angle = cv2.minAreaRect(coords)[-1]
####        if angle < -45:
####            angle = -(90 + angle)
####        else:
####            angle = -angle
####        (h, w) = image.shape[:2]
####        center = (w // 2, h // 2)
####        M = cv2.getRotationMatrix2D(center, angle, 1.0)
####        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
####
####    image = cv2.rotate(image, cv2.ROTATE_180)
####    return image
####
####st.title("OCR Intelligent - EasyOCR (Français)")
####uploaded_file = st.file_uploader("Charge une image de document", type=["jpg", "jpeg", "png"])
####
####if uploaded_file:
####    original_image = Image.open(uploaded_file)
####    st.image(original_image, caption="Image originale", use_container_width=True)
####
####    processed_image = preprocess_image(original_image)
####
####    st.subheader("Zones détectées")
####    result_with_boxes = reader.readtext(processed_image)
####    image_with_boxes = processed_image.copy()
####
####    for (bbox, text, conf) in result_with_boxes:
####        (top_left, top_right, bottom_right, bottom_left) = bbox
####        top_left = tuple(map(int, top_left))
####        bottom_right = tuple(map(int, bottom_right))
####        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
####        cv2.putText(image_with_boxes, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
####
####    st.image(image_with_boxes, caption="Zones de texte détectées", use_container_width=True)
####
####    st.subheader("Texte extrait et corrections")
####    line_texts = [text for (_, text, _) in result_with_boxes if text is not None]
####    raw_text = "\n".join(line_texts)
####    words = raw_text.split()
####
####    corrected_table = []
####    updated = False
####    for idx, word in enumerate(words):
####        base = word.strip()
####        if not base.isalpha():
####            continue
####        auto_corrected = spell.correction(base)
####        previous_manual = manual_corrections.get(base, "")
####        manual = st.text_input(f"Correction manuelle pour : {base}", value=previous_manual or auto_corrected, key=f"manual_{base}_{idx}")
####        if manual != previous_manual:
####            manual_corrections[base] = manual
####            updated = True
####        corrected_table.append({"Mot extrait": base, "Correction auto": auto_corrected, "Correction manuelle": manual})
####
####    df = pd.DataFrame(corrected_table)
####    st.dataframe(df, use_container_width=True)
####
####    if updated:
####        save_corrections(manual_corrections)
####        st.success("Corrections sauvegardées.")





######import streamlit as st
######import numpy as np
######import cv2
######import pandas as pd
######import json
######import os
######from PIL import Image
######import easyocr
######from spellchecker import SpellChecker
######
######st.set_page_config(page_title="OCR Intelligent FR avec EasyOCR", layout="centered")
######
######@st.cache_resource
######def load_easyocr_model():
######    return easyocr.Reader(['fr'], gpu=False)
######
######reader = load_easyocr_model()
######spell = SpellChecker(language='fr')
######DB_FILE = "corrections.json"
######
####### Charger ou initialiser base de corrections manuelles
######def load_corrections():
######    if os.path.exists(DB_FILE):
######        with open(DB_FILE, "r", encoding="utf-8") as f:
######            return json.load(f)
######    return {}
######
####### Sauvegarder corrections
######def save_corrections(corrections):
######    with open(DB_FILE, "w", encoding="utf-8") as f:
######        json.dump(corrections, f, ensure_ascii=False, indent=2)
######
######manual_corrections = load_corrections()
######
####### Prétraitement image
######def preprocess_image(image_pil):
######    image = np.array(image_pil.convert("RGB"))
######    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
######    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
######    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
######    enhanced = clahe.apply(denoised)
######
######    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
######    edged = cv2.Canny(blurred, 50, 150)
######    coords = np.column_stack(np.where(edged > 0))
######    if coords.shape[0] > 0:
######        angle = cv2.minAreaRect(coords)[-1]
######        if angle < -45:
######            angle = -(90 + angle)
######        else:
######            angle = -angle
######        (h, w) = image.shape[:2]
######        center = (w // 2, h // 2)
######        M = cv2.getRotationMatrix2D(center, angle, 1.0)
######        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
######
######    image = cv2.rotate(image, cv2.ROTATE_180)
######    return image
######
######st.title("OCR Intelligent - EasyOCR (Français)")
######uploaded_file = st.file_uploader("Charge une image de document", type=["jpg", "jpeg", "png"])
######
######if uploaded_file:
######    original_image = Image.open(uploaded_file)
######    st.image(original_image, caption="Image originale", use_container_width=True)
######
######    processed_image = preprocess_image(original_image)
######
######    st.subheader("Zones détectées")
######    result_with_boxes = reader.readtext(processed_image)
######    image_with_boxes = processed_image.copy()
######
######    for (bbox, text, conf) in result_with_boxes:
######        (top_left, top_right, bottom_right, bottom_left) = bbox
######        top_left = tuple(map(int, top_left))
######        bottom_right = tuple(map(int, bottom_right))
######        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
######        cv2.putText(image_with_boxes, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
######
######    st.image(image_with_boxes, caption="Zones de texte détectées", use_container_width=True)
######
######    st.subheader("Texte extrait brut")
######    line_texts = [text for (_, text, _) in result_with_boxes if text is not None]
######    raw_text = "\n".join(line_texts)
######    edited_text = st.text_area("Édite le texte brut ici pour fusionner ou corriger des mots avant analyse :", raw_text, height=200)
######
######    st.subheader("Tableau des corrections mot par mot")
######    words = edited_text.split()
######    corrected_table = []
######    updated = False
######    for idx, word in enumerate(words):
######        base = word.strip()
######        if not base.isalpha():
######            continue
######        auto_corrected = spell.correction(base)
######        previous_manual = manual_corrections.get(base, "")
######        manual = st.text_input(f"Correction manuelle pour : {base}", value=previous_manual or auto_corrected, key=f"manual_{base}_{idx}")
######        if manual != previous_manual:
######            manual_corrections[base] = manual
######            updated = True
######        corrected_table.append({"Mot extrait": base, "Correction auto": auto_corrected, "Correction manuelle": manual})
######
######    df = pd.DataFrame(corrected_table)
######    st.dataframe(df, use_container_width=True)
######
######    if updated:
######        save_corrections(manual_corrections)
######        st.success("Corrections sauvegardées.")





###########import streamlit as st
###########import numpy as np
###########import cv2
###########import pandas as pd
###########import json
###########import os
###########from PIL import Image
###########import easyocr
###########from spellchecker import SpellChecker
###########
###########st.set_page_config(page_title="OCR Intelligent FR avec EasyOCR", layout="centered")
###########
###########@st.cache_resource
###########def load_easyocr_model():
###########    return easyocr.Reader(['fr'], gpu=False)
###########
###########reader = load_easyocr_model()
###########spell = SpellChecker(language='fr')
###########DB_FILE = "corrections.json"
###########
############ Charger ou initialiser base de corrections manuelles
###########def load_corrections():
###########    if os.path.exists(DB_FILE):
###########        with open(DB_FILE, "r", encoding="utf-8") as f:
###########            return json.load(f)
###########    return {}
###########
############ Sauvegarder corrections
###########def save_corrections(corrections):
###########    with open(DB_FILE, "w", encoding="utf-8") as f:
###########        json.dump(corrections, f, ensure_ascii=False, indent=2)
###########
###########manual_corrections = load_corrections()
###########
############ Prétraitement image
###########def preprocess_image(image_pil):
###########    image = np.array(image_pil.convert("RGB"))
###########    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
###########    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
###########    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
###########    enhanced = clahe.apply(denoised)
###########
###########    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
###########    edged = cv2.Canny(blurred, 50, 150)
###########    coords = np.column_stack(np.where(edged > 0))
###########    if coords.shape[0] > 0:
###########        angle = cv2.minAreaRect(coords)[-1]
###########        if angle < -45:
###########            angle = -(90 + angle)
###########        else:
###########            angle = -angle
###########        (h, w) = image.shape[:2]
###########        center = (w // 2, h // 2)
###########        M = cv2.getRotationMatrix2D(center, angle, 1.0)
###########        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
###########
###########    image = cv2.rotate(image, cv2.ROTATE_180)
###########    return image
###########
###########st.title("OCR Intelligent - EasyOCR (Français)")
###########uploaded_file = st.file_uploader("Charge une image de document", type=["jpg", "jpeg", "png"])
###########
###########if uploaded_file:
###########    original_image = Image.open(uploaded_file)
###########    st.image(original_image, caption="Image originale", use_container_width=True)
###########
###########    processed_image = preprocess_image(original_image)
###########
###########    st.subheader("Zones détectées")
###########    result_with_boxes = reader.readtext(processed_image, detail=1, paragraph=False)
###########    image_with_boxes = processed_image.copy()
###########
###########    for (bbox, text, conf) in result_with_boxes:
###########        (top_left, top_right, bottom_right, bottom_left) = bbox
###########        top_left = tuple(map(int, top_left))
###########        bottom_right = tuple(map(int, bottom_right))
###########        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
###########        cv2.putText(image_with_boxes, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
###########
###########    st.image(image_with_boxes, caption="Zones de texte détectées", use_container_width=True)
###########
###########    # OCR brut sans zones pour extraire tout le texte même non détecté dans les boîtes
###########    full_text_result = reader.readtext(processed_image, detail=0, paragraph=True)
###########    full_text_raw = "\n".join(full_text_result)
###########
###########    st.subheader("Texte extrait brut complet")
###########    edited_text = st.text_area("Édite le texte brut ici pour fusionner ou corriger des mots avant analyse :", full_text_raw, height=200)
###########
###########    st.subheader("Tableau des corrections mot par mot")
###########    words = edited_text.split()
###########    corrected_table = []
###########    updated = False
###########
###########    st.markdown("**Ajoute un mot manuellement si EasyOCR l'a oublié :**")
###########    new_word = st.text_input("Mot à ajouter", "")
###########    if new_word.strip():
###########        words.append(new_word.strip())
###########
###########    for idx, word in enumerate(words):
###########        base = word.strip()
###########        if not base.isalpha():
###########            continue
###########        auto_corrected = spell.correction(base)
###########        previous_manual = manual_corrections.get(base, "")
###########        manual = st.text_input(f"Correction manuelle pour : {base}", value=previous_manual or auto_corrected, key=f"manual_{base}_{idx}")
###########        if manual != previous_manual:
###########            manual_corrections[base] = manual
###########            updated = True
###########        corrected_table.append({"Mot extrait": base, "Correction auto": auto_corrected, "Correction manuelle": manual})
###########
###########    df = pd.DataFrame(corrected_table)
###########    st.dataframe(df, use_container_width=True)
###########
###########    if updated:
###########        save_corrections(manual_corrections)
###########        st.success("Corrections sauvegardées.")




########import streamlit as st
########import numpy as np
########import cv2
########import pandas as pd
########import json
########import os
########from PIL import Image
########import easyocr
########from spellchecker import SpellChecker
########
########st.set_page_config(page_title="OCR Intelligent FR avec EasyOCR", layout="centered")
########
########@st.cache_resource
########def load_easyocr_model():
########    return easyocr.Reader(['fr'], gpu=False)
########
########reader = load_easyocr_model()
########spell = SpellChecker(language='fr')
########DB_FILE = "corrections.json"
########
######### Charger ou initialiser base de corrections manuelles
########def load_corrections():
########    if os.path.exists(DB_FILE):
########        with open(DB_FILE, "r", encoding="utf-8") as f:
########            return json.load(f)
########    return {}
########
######### Sauvegarder corrections
########def save_corrections(corrections):
########    with open(DB_FILE, "w", encoding="utf-8") as f:
########        json.dump(corrections, f, ensure_ascii=False, indent=2)
########
########manual_corrections = load_corrections()
########
######### Prétraitement image
########def preprocess_image(image_pil):
########    image = np.array(image_pil.convert("RGB"))
########    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
########    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
########    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
########    enhanced = clahe.apply(denoised)
########
########    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
########    edged = cv2.Canny(blurred, 50, 150)
########    coords = np.column_stack(np.where(edged > 0))
########    if coords.shape[0] > 0:
########        angle = cv2.minAreaRect(coords)[-1]
########        if angle < -45:
########            angle = -(90 + angle)
########        else:
########            angle = -angle
########        (h, w) = image.shape[:2]
########        center = (w // 2, h // 2)
########        M = cv2.getRotationMatrix2D(center, angle, 1.0)
########        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
########
########    image = cv2.rotate(image, cv2.ROTATE_180)
########    return image
########
########st.title("OCR Intelligent - EasyOCR (Français)")
########uploaded_file = st.file_uploader("Charge une image de document", type=["jpg", "jpeg", "png"])
########
########if uploaded_file:
########    original_image = Image.open(uploaded_file)
########    st.image(original_image, caption="Image originale", use_container_width=True)
########
########    processed_image = preprocess_image(original_image)
########
########    st.subheader("Zones détectées automatiquement")
########    result_with_boxes = reader.readtext(processed_image, detail=1, paragraph=False)
########    image_with_boxes = processed_image.copy()
########
########    for (bbox, text, conf) in result_with_boxes:
########        (top_left, top_right, bottom_right, bottom_left) = bbox
########        top_left = tuple(map(int, top_left))
########        bottom_right = tuple(map(int, bottom_right))
########        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
########        cv2.putText(image_with_boxes, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
########
########    # Ajout de zones personnalisées par l'utilisateur
########    st.subheader("Ajouter un encadrement manuel")
########    x = st.number_input("X (gauche)", min_value=0, value=0)
########    y = st.number_input("Y (haut)", min_value=0, value=0)
########    w = st.number_input("Largeur", min_value=1, value=100)
########    h = st.number_input("Hauteur", min_value=1, value=30)
########    add_box = st.button("Ajouter cette zone encadrée")
########
########    manual_boxes = []
########    if "manual_boxes" not in st.session_state:
########        st.session_state.manual_boxes = []
########
########    if add_box:
########        st.session_state.manual_boxes.append(((x, y), (x + w, y + h)))
########
########    for (pt1, pt2) in st.session_state.manual_boxes:
########        cv2.rectangle(image_with_boxes, pt1, pt2, (255, 0, 255), 2)
########
########    st.image(image_with_boxes, caption="Zones détectées (automatiques + manuelles)", use_container_width=True)
########
########    # OCR brut sans zones pour extraire tout le texte même non détecté dans les boîtes
########    full_text_result = reader.readtext(processed_image, detail=0, paragraph=True)
########    full_text_raw = "\n".join(full_text_result)
########
########    st.subheader("Texte extrait brut complet")
########    edited_text = st.text_area("Édite le texte brut ici pour fusionner ou corriger des mots avant analyse :", full_text_raw, height=200)
########
########    st.subheader("Tableau des corrections mot par mot")
########    words = edited_text.split()
########    corrected_table = []
########    updated = False
########
########    st.markdown("**Ajoute un mot manuellement si EasyOCR l'a oublié :**")
########    new_word = st.text_input("Mot à ajouter", "")
########    if new_word.strip():
########        words.append(new_word.strip())
########
########    for idx, word in enumerate(words):
########        base = word.strip()
########        if not base.isalpha():
########            continue
########        auto_corrected = spell.correction(base)
########        previous_manual = manual_corrections.get(base, "")
########        manual = st.text_input(f"Correction manuelle pour : {base}", value=previous_manual or auto_corrected, key=f"manual_{base}_{idx}")
########        if manual != previous_manual:
########            manual_corrections[base] = manual
########            updated = True
########        corrected_table.append({"Mot extrait": base, "Correction auto": auto_corrected, "Correction manuelle": manual})
########
########    df = pd.DataFrame(corrected_table)
########    st.dataframe(df, use_container_width=True)
########
########    if updated:
########        save_corrections(manual_corrections)
########        st.success("Corrections sauvegardées.")




#####import streamlit as st
#####import numpy as np
#####import cv2
#####import pandas as pd
#####import json
#####import os
#####from PIL import Image
#####import easyocr
#####from spellchecker import SpellChecker
#####
#####st.set_page_config(page_title="OCR Intelligent FR avec EasyOCR", layout="centered")
#####
#####@st.cache_resource
#####def load_easyocr_model():
#####    return easyocr.Reader(['fr'], gpu=False)
#####
#####reader = load_easyocr_model()
#####spell = SpellChecker(language='fr')
#####DB_FILE = "corrections.json"
#####
###### Charger ou initialiser base de corrections manuelles
#####def load_corrections():
#####    if os.path.exists(DB_FILE):
#####        with open(DB_FILE, "r", encoding="utf-8") as f:
#####            return json.load(f)
#####    return {}
#####
###### Sauvegarder corrections
#####def save_corrections(corrections):
#####    with open(DB_FILE, "w", encoding="utf-8") as f:
#####        json.dump(corrections, f, ensure_ascii=False, indent=2)
#####
#####manual_corrections = load_corrections()
#####
#####def preprocess_image(image_pil, crop_box=None, zoom_level=1.0):
#####    image = np.array(image_pil.convert("RGB"))
#####
#####    if crop_box:
#####        x, y, w, h = crop_box
#####        image = image[y:y + h, x:x + w]
#####
#####    if zoom_level != 1.0:
#####        h, w = image.shape[:2]
#####        new_size = (int(w * zoom_level), int(h * zoom_level))
#####        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
#####
#####    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#####    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
#####    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#####    enhanced = clahe.apply(denoised)
#####
#####    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
#####    edged = cv2.Canny(blurred, 50, 150)
#####    coords = np.column_stack(np.where(edged > 0))
#####    if coords.shape[0] > 0:
#####        angle = cv2.minAreaRect(coords)[-1]
#####        if angle < -45:
#####            angle = -(90 + angle)
#####        else:
#####            angle = -angle
#####        (h, w) = image.shape[:2]
#####        center = (w // 2, h // 2)
#####        M = cv2.getRotationMatrix2D(center, angle, 1.0)
#####        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#####
#####    image = cv2.rotate(image, cv2.ROTATE_180)
#####    return image
#####
#####st.title("OCR Intelligent - EasyOCR (Français)")
#####uploaded_file = st.file_uploader("Charge une image de document", type=["jpg", "jpeg", "png"])
#####
#####if uploaded_file:
#####    original_image = Image.open(uploaded_file)
#####    st.image(original_image, caption="Image originale", use_container_width=True)
#####
#####    st.subheader("Encadrement manuel pour améliorer la rotation")
#####    col1, col2 = st.columns(2)
#####    with col1:
#####        x_rot = st.number_input("X rotation", min_value=0, value=0)
#####        y_rot = st.number_input("Y rotation", min_value=0, value=0)
#####    with col2:
#####        w_rot = st.number_input("Largeur rotation", min_value=1, value=original_image.width)
#####        h_rot = st.number_input("Hauteur rotation", min_value=1, value=original_image.height)
#####
#####    use_manual_rotation = st.checkbox("Utiliser ce cadrage pour la rotation ?")
#####    crop_box = (x_rot, y_rot, w_rot, h_rot) if use_manual_rotation else None
#####
#####    st.subheader("Zoom manuel (1.0 = 100%)")
#####    zoom_level = st.slider("Niveau de zoom", 0.5, 3.0, 1.0, 0.1)
#####
#####    processed_image = preprocess_image(original_image, crop_box, zoom_level)
#####
#####    st.subheader("Ajout de zones manuelles à OCR")
#####    if "manual_ocr_zones" not in st.session_state:
#####        st.session_state.manual_ocr_zones = []
#####
#####    x_m = st.number_input("X zone manuelle", min_value=0, value=0, key="x_m")
#####    y_m = st.number_input("Y zone manuelle", min_value=0, value=0, key="y_m")
#####    w_m = st.number_input("Largeur zone manuelle", min_value=1, value=100, key="w_m")
#####    h_m = st.number_input("Hauteur zone manuelle", min_value=1, value=30, key="h_m")
#####    if st.button("Ajouter zone manuelle"):
#####        st.session_state.manual_ocr_zones.append((x_m, y_m, w_m, h_m))
#####
#####    st.subheader("Zones détectées automatiquement")
#####    result_with_boxes = reader.readtext(processed_image, detail=1, paragraph=False)
#####    image_with_boxes = processed_image.copy()
#####
#####    for (bbox, text, conf) in result_with_boxes:
#####        (top_left, top_right, bottom_right, bottom_left) = bbox
#####        top_left = tuple(map(int, top_left))
#####        bottom_right = tuple(map(int, bottom_right))
#####        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
#####        cv2.putText(image_with_boxes, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
#####
#####    manual_texts = []
#####    for i, (x, y, w, h) in enumerate(st.session_state.manual_ocr_zones):
#####        roi = processed_image[y:y + h, x:x + w]
#####        manual_results = reader.readtext(roi, detail=0)
#####        manual_texts.extend(manual_results)
#####        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (255, 0, 255), 2)
#####
#####    st.image(image_with_boxes, caption="Zones détectées (automatiques + manuelles)", use_container_width=True)
#####
#####    all_text = [text for (_, text, _) in result_with_boxes] + manual_texts
#####    full_text_raw = "\n".join(all_text)
#####
#####    st.subheader("Texte extrait brut complet")
#####    edited_text = st.text_area("Édite le texte brut ici pour fusionner ou corriger des mots avant analyse :", full_text_raw, height=200)
#####
#####    st.subheader("Ajout de mots manuellement (séparés par des virgules)")
#####    manual_input = st.text_input("Ajouter plusieurs mots manuellement :", "")
#####    if manual_input:
#####        added_words = [w.strip() for w in manual_input.split(",") if w.strip()]
#####    else:
#####        added_words = []
#####
#####    all_words = edited_text.split() + added_words
#####
#####    corrected_table = []
#####    updated = False
#####
#####    for idx, word in enumerate(all_words):
#####        base = word.strip()
#####        if not base.isalpha():
#####            continue
#####        auto_corrected = spell.correction(base)
#####        previous_manual = manual_corrections.get(base, "")
#####        manual = st.text_input(f"Correction manuelle pour : {base}", value=previous_manual or auto_corrected, key=f"manual_{base}_{idx}")
#####        if manual != previous_manual:
#####            manual_corrections[base] = manual
#####            updated = True
#####        corrected_table.append({"Mot extrait": base, "Correction auto": auto_corrected, "Correction manuelle": manual})
#####
#####    df = pd.DataFrame(corrected_table)
#####    st.dataframe(df, use_container_width=True)
#####
#####    if updated:
#####        save_corrections(manual_corrections)
#####        st.success("Corrections sauvegardées.")







####import streamlit as st
####import numpy as np
####import cv2
####import pandas as pd
####import json
####import os
####from PIL import Image
####import easyocr
####from spellchecker import SpellChecker
####from io import BytesIO
####
####st.set_page_config(page_title="OCR Intelligent FR avec EasyOCR", layout="centered")
####
####@st.cache_resource
####def load_easyocr_model():
####    return easyocr.Reader(['fr'], gpu=False)
####
####reader = load_easyocr_model()
####spell = SpellChecker(language='fr')
####DB_FILE = "corrections.json"
####
##### Charger ou initialiser base de corrections manuelles
####def load_corrections():
####    if os.path.exists(DB_FILE):
####        with open(DB_FILE, "r", encoding="utf-8") as f:
####            return json.load(f)
####    return {}
####
####def save_corrections(corrections):
####    with open(DB_FILE, "w", encoding="utf-8") as f:
####        json.dump(corrections, f, ensure_ascii=False, indent=2)
####
####manual_corrections = load_corrections()
####
####
####def preprocess_image(image_pil, crop_box=None, zoom_level=1.0):
####    image = np.array(image_pil.convert("RGB"))
####    if crop_box:
####        x, y, w, h = crop_box
####        image = image[y:y + h, x:x + w]
####    if zoom_level != 1.0:
####        h, w = image.shape[:2]
####        new_size = (int(w * zoom_level), int(h * zoom_level))
####        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
####    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
####    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
####    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
####    enhanced = clahe.apply(denoised)
####    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
####    edged = cv2.Canny(blurred, 50, 150)
####    coords = np.column_stack(np.where(edged > 0))
####    if coords.shape[0] > 0:
####        angle = cv2.minAreaRect(coords)[-1]
####        angle = -(90 + angle) if angle < -45 else -angle
####        (h, w) = image.shape[:2]
####        center = (w // 2, h // 2)
####        M = cv2.getRotationMatrix2D(center, angle, 1.0)
####        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
####    image = cv2.rotate(image, cv2.ROTATE_180)
####    return image
####
####st.title("OCR Intelligent - EasyOCR (Français)")
####uploaded_file = st.file_uploader("Charge une image de document", type=["jpg", "jpeg", "png"])
####
####if uploaded_file:
####    original_image = Image.open(uploaded_file)
####    st.image(original_image, caption="Image originale", use_container_width=True)
####
####    st.subheader("Encadrement manuel pour améliorer la rotation")
####    col1, col2 = st.columns(2)
####    with col1:
####        x_rot = st.number_input("X rotation", min_value=0, value=0)
####        y_rot = st.number_input("Y rotation", min_value=0, value=0)
####    with col2:
####        w_rot = st.number_input("Largeur rotation", min_value=1, value=original_image.width)
####        h_rot = st.number_input("Hauteur rotation", min_value=1, value=original_image.height)
####
####    use_manual_rotation = st.checkbox("Utiliser ce cadrage pour la rotation ?")
####    crop_box = (x_rot, y_rot, w_rot, h_rot) if use_manual_rotation else None
####
####    st.subheader("Zoom manuel (1.0 = 100%)")
####    zoom_level = st.slider("Niveau de zoom", 0.5, 3.0, 1.0, 0.1)
####
####    processed_image = preprocess_image(original_image, crop_box, zoom_level)
####    st.subheader("Zones détectées automatiquement")
####    result_with_boxes = reader.readtext(processed_image, detail=1, paragraph=False)
####    image_with_boxes = processed_image.copy()
####
####    for (bbox, text, conf) in result_with_boxes:
####        (top_left, top_right, bottom_right, bottom_left) = bbox
####        top_left = tuple(map(int, top_left))
####        bottom_right = tuple(map(int, bottom_right))
####        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
####        cv2.putText(image_with_boxes, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
####
####    st.image(image_with_boxes, caption="Zones détectées automatiquement", use_container_width=True)
####
####    all_text = [text for (_, text, _) in result_with_boxes]
####    full_text_raw = "\n".join(all_text)
####
####    st.subheader("Texte extrait brut complet")
####    edited_text = st.text_area("Texte brut (modifiable avant analyse) :", full_text_raw, height=200)
####
####    st.subheader("Ajout manuel de mots (séparés par des virgules)")
####    manual_input = st.text_input("Ajouter des mots :", "")
####    if manual_input:
####        new_manuals = [w.strip() for w in manual_input.split(",") if w.strip()]
####        for word in new_manuals:
####            if word not in manual_corrections:
####                manual_corrections[word] = spell.correction(word)
####        save_corrections(manual_corrections)
####        st.success("Ajouts enregistrés")
####
####    all_words = edited_text.split() + list(manual_corrections.keys())
####    all_words = list(dict.fromkeys(all_words))  # Remove duplicates while preserving order
####
####    corrected_table = []
####    updated = False
####    for idx, word in enumerate(all_words):
####        base = word.strip()
####        if not base:
####            continue
####        auto = spell.correction(base)
####        manual = st.text_input(f"Correction manuelle : {base}", value=manual_corrections.get(base, auto), key=f"manual_{base}_{idx}")
####        if manual != manual_corrections.get(base, auto):
####            manual_corrections[base] = manual
####            updated = True
####        delete = st.button("❌ Supprimer", key=f"delete_{base}_{idx}")
####        if delete:
####            if base in manual_corrections:
####                del manual_corrections[base]
####                updated = True
####            continue
####        corrected_table.append({"Mot extrait": base, "Correction auto": auto, "Correction manuelle": manual})
####
####    df = pd.DataFrame(corrected_table)
####    st.dataframe(df, use_container_width=True)
####
####    if updated:
####        save_corrections(manual_corrections)
####        st.success("Corrections sauvegardées")
####
####    st.subheader("\U0001F4E5 Exporter vers Excel uniquement les mots vérifiés")
####    if st.button("\U0001F4E4 Télécharger le tableau corrigé"):
####        export_df = pd.DataFrame([
####            row for row in corrected_table if row["Mot extrait"] in manual_corrections
####        ])
####        output = BytesIO()
####        export_df.to_excel(output, index=False, engine='openpyxl')
####        st.download_button(
####            label="\U0001F4C4 Télécharger le fichier Excel",
####            data=output.getvalue(),
####            file_name="tableau_corrige.xlsx",
####            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
####        )




###import streamlit as st
###import numpy as np
###import cv2
###import pandas as pd
###import json
###import os
###from PIL import Image
###import easyocr
###from spellchecker import SpellChecker
###from io import BytesIO
###
###st.set_page_config(page_title="OCR Intelligent FR avec EasyOCR", layout="centered")
###
###@st.cache_resource
###def load_easyocr_model():
###    return easyocr.Reader(['fr'], gpu=False)
###
###reader = load_easyocr_model()
###spell = SpellChecker(language='fr')
###DB_FILE = "corrections.json"
###STRUCT_FILE = "structure.json"
###
#### Charger ou initialiser base de corrections manuelles
###def load_corrections():
###    if os.path.exists(DB_FILE):
###        with open(DB_FILE, "r", encoding="utf-8") as f:
###            return json.load(f)
###    return {}
###
###def save_corrections(corrections):
###    with open(DB_FILE, "w", encoding="utf-8") as f:
###        json.dump(corrections, f, ensure_ascii=False, indent=2)
###
#### Charger ou initialiser la structure Excel
###def load_structure():
###    if os.path.exists(STRUCT_FILE):
###        with open(STRUCT_FILE, "r", encoding="utf-8") as f:
###            return json.load(f)
###    return {}
###
###def save_structure(structure):
###    with open(STRUCT_FILE, "w", encoding="utf-8") as f:
###        json.dump(structure, f, ensure_ascii=False, indent=2)
###
###manual_corrections = load_corrections()
###word_structure = load_structure()
###
###if 'added_words' not in st.session_state:
###    st.session_state.added_words = []
###
###if 'deleted_words' not in st.session_state:
###    st.session_state.deleted_words = set()
###
###def preprocess_image(image_pil, manual_angle=0):
###    image = np.array(image_pil.convert("RGB"))
###    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
###    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
###    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
###    enhanced = clahe.apply(denoised)
###    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
###    edged = cv2.Canny(blurred, 50, 150)
###    coords = np.column_stack(np.where(edged > 0))
###    if coords.shape[0] > 0:
###        angle = cv2.minAreaRect(coords)[-1]
###        angle = -(90 + angle) if angle < -45 else -angle
###        (h, w) = image.shape[:2]
###        center = (w // 2, h // 2)
###        M = cv2.getRotationMatrix2D(center, angle, 1.0)
###        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
###
###    # Apply manual rotation if any
###    if manual_angle != 0:
###        (h, w) = image.shape[:2]
###        center = (w // 2, h // 2)
###        M_manual = cv2.getRotationMatrix2D(center, manual_angle, 1.0)
###        image = cv2.warpAffine(image, M_manual, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
###
###    return image
###
###st.title("OCR Intelligent - EasyOCR (Français)")
###uploaded_file = st.file_uploader("Charge une image de document", type=["jpg", "jpeg", "png"])
###
###if uploaded_file:
###    original_image = Image.open(uploaded_file)
###    st.image(original_image, caption="Image originale", use_container_width=True)
###
###    angle_manual = st.slider("Rotation manuelle (degrés)", -180, 180, 0)
###    processed_image = preprocess_image(original_image, manual_angle=angle_manual)
###
###    result_with_boxes = reader.readtext(processed_image, detail=1, paragraph=False)
###    image_with_boxes = processed_image.copy()
###
###    for (bbox, text, conf) in result_with_boxes:
###        (top_left, top_right, bottom_right, bottom_left) = bbox
###        top_left = tuple(map(int, top_left))
###        bottom_right = tuple(map(int, bottom_right))
###        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
###        cv2.putText(image_with_boxes, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
###
###    st.image(image_with_boxes, caption="Zones détectées automatiquement", use_container_width=True)
###
###    all_text = [text for (_, text, _) in result_with_boxes]
###    full_text_raw = "\n".join(all_text)
###
###    st.subheader("Texte extrait brut complet")
###    edited_text = st.text_area("Texte brut (modifiable avant analyse) :", full_text_raw, height=200)
###
###    st.subheader("Ajout manuel de mots (séparés par des virgules)")
###    manual_input = st.text_input("Ajouter des mots :", "")
###    if manual_input:
###        new_manuals = [w.strip() for w in manual_input.split(",") if w.strip()]
###        for word in new_manuals:
###            if word not in st.session_state.added_words:
###                st.session_state.added_words.append(word)
###                manual_corrections[word] = spell.correction(word)
###        save_corrections(manual_corrections)
###        st.success("Ajouts enregistrés")
###
###    all_words = edited_text.split() + st.session_state.added_words
###    all_words = list(dict.fromkeys(all_words))
###
###    st.subheader("Corrections et placement dans le tableau")
###    corrected_table = []
###    updated = False
###    for idx, word in enumerate(all_words):
###        base = word.strip()
###        if not base or base in st.session_state.deleted_words:
###            continue
###        auto = spell.correction(base)
###        manual = st.text_input(f"Correction manuelle : {base}", value=manual_corrections.get(base, auto), key=f"manual_{base}_{idx}")
###        if manual != manual_corrections.get(base, auto):
###            manual_corrections[base] = manual
###            updated = True
###        if st.button(f"❌ Supprimer {base}", key=f"delete_{base}_{idx}"):
###            st.session_state.deleted_words.add(base)
###            if base in st.session_state.added_words:
###                st.session_state.added_words.remove(base)
###            if base in manual_corrections:
###                del manual_corrections[base]
###            updated = True
###            continue
###
###        # Structure positionnelle pour Excel
###        if manual not in word_structure:
###            st.markdown(f"**Associer la position dans Excel pour '{manual}'**")
###            col1, col2 = st.columns(2)
###            with col1:
###                row = st.number_input(f"Ligne pour '{manual}'", min_value=1, value=1, key=f"row_{manual}_{idx}")
###            with col2:
###                col = st.number_input(f"Colonne pour '{manual}'", min_value=1, value=1, key=f"col_{manual}_{idx}")
###            if st.button(f"✅ Enregistrer position de {manual}", key=f"save_pos_{manual}_{idx}"):
###                word_structure[manual] = [row, col]
###                save_structure(word_structure)
###                st.success(f"Position enregistrée pour {manual}")
###
###        corrected_table.append({"Mot extrait": base, "Correction auto": auto, "Correction manuelle": manual})
###
###    df = pd.DataFrame(corrected_table)
###    st.dataframe(df, use_container_width=True)
###
###    if updated:
###        save_corrections(manual_corrections)
###        st.success("Corrections sauvegardées")
###
###    st.subheader("\U0001F4E5 Exporter vers Excel structuré")
###    if st.button("\U0001F4E4 Télécharger le tableau structuré"):
###        max_row = max([pos[0] for pos in word_structure.values()]) if word_structure else 10
###        max_col = max([pos[1] for pos in word_structure.values()]) if word_structure else 10
###        excel_matrix = [["" for _ in range(max_col)] for _ in range(max_row)]
###
###        for entry in corrected_table:
###            mot = entry["Correction manuelle"]
###            if mot in word_structure:
###                r, c = word_structure[mot]
###                excel_matrix[r-1][c-1] = mot
###
###        df_struct = pd.DataFrame(excel_matrix)
###        output = BytesIO()
###        df_struct.to_excel(output, index=False, header=False, engine='openpyxl')
###        st.download_button(
###            label="\U0001F4C4 Télécharger l'Excel structuré",
###            data=output.getvalue(),
###            file_name="tableau_structure.xlsx",
###            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
###        )
###
###





import streamlit as st
import numpy as np
import cv2
import pandas as pd
import json
import os
from PIL import Image
import easyocr
from spellchecker import SpellChecker
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="Inventaire à jour", layout="centered")

@st.cache_resource
def load_easyocr_model():
    return easyocr.Reader(['fr'], gpu=False)

reader = load_easyocr_model()
spell = SpellChecker(language='fr')
DB_FILE = "corrections.json"
STRUCT_FILE = "structure.json"
TEMPLATE_FILE = "Inventaire à jour.xlsx"

def load_corrections():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_corrections(corrections):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)

def load_structure():
    if os.path.exists(STRUCT_FILE):
        with open(STRUCT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_structure(structure):
    with open(STRUCT_FILE, "w", encoding="utf-8") as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)

manual_corrections = load_corrections()
word_structure = load_structure()

if 'added_words' not in st.session_state:
    st.session_state.added_words = []

if 'deleted_words' not in st.session_state:
    st.session_state.deleted_words = set()

def preprocess_image(image_pil, manual_angle=0):
    image = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    coords = np.column_stack(np.where(edged > 0))
    if coords.shape[0] > 0:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if manual_angle != 0:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M_manual = cv2.getRotationMatrix2D(center, manual_angle, 1.0)
        image = cv2.warpAffine(image, M_manual, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return image

st.title("Inventaire à jour")
uploaded_file = st.file_uploader("Charge une image de document", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Image originale", use_container_width=True)

    angle_manual = st.slider("Rotation manuelle (degrés)", -180, 180, 0)
    processed_image = preprocess_image(original_image, manual_angle=angle_manual)

    result_with_boxes = reader.readtext(processed_image, detail=1, paragraph=False)
    image_with_boxes = processed_image.copy()

    for (bbox, text, conf) in result_with_boxes:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image_with_boxes, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    st.image(image_with_boxes, caption="Zones détectées automatiquement", use_container_width=True)

    all_text = [text for (_, text, _) in result_with_boxes]
    full_text_raw = "\n".join(all_text)

    st.subheader("Texte extrait brut complet")
    edited_text = st.text_area("Texte brut (modifiable avant analyse) :", full_text_raw, height=200)

    st.subheader("Ajout manuel de mots (séparés par des virgules)")
    manual_input = st.text_input("Ajouter des mots :", "")
    if manual_input:
        new_manuals = [w.strip() for w in manual_input.split(",") if w.strip()]
        for word in new_manuals:
            if word not in st.session_state.added_words:
                st.session_state.added_words.append(word)
                manual_corrections[word] = spell.correction(word)
        save_corrections(manual_corrections)
        st.success("Ajouts enregistrés")

    all_words = edited_text.split() + st.session_state.added_words

    st.subheader("Corrections et placement dans le tableau")
    corrected_table = []
    updated = False
    for idx, word in enumerate(all_words):
        base = word.strip()
        if not base or base in st.session_state.deleted_words:
            continue
        auto = spell.correction(base)
        manual = st.text_input(f"Correction manuelle : {base}", value=manual_corrections.get(base, auto), key=f"manual_{base}_{idx}")

        default_row, default_col = word_structure.get(manual, [1, 1])
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**Position dans le tableau Excel :**")
        with col2:
            row = st.number_input("Ligne", min_value=1, value=default_row, key=f"row_{base}_{idx}")
        with col3:
            col = st.number_input("Colonne", min_value=1, value=default_col, key=f"col_{base}_{idx}")

        if st.button(f"💾 Sauvegarder position pour {manual}", key=f"save_pos_{manual}_{idx}"):
            word_structure[manual] = [row, col]
            save_structure(word_structure)
            st.success(f"Position enregistrée pour {manual}")

        if manual != manual_corrections.get(base, auto):
            manual_corrections[base] = manual
            updated = True

        if st.button(f"❌ Supprimer {base}", key=f"delete_{base}_{idx}"):
            st.session_state.deleted_words.add(base)
            if base in st.session_state.added_words:
                st.session_state.added_words.remove(base)
            if base in manual_corrections:
                del manual_corrections[base]
            updated = True
            continue

        corrected_table.append({"Mot extrait": base, "Correction auto": auto, "Correction manuelle": manual, "Ligne": row, "Colonne": col})

    df = pd.DataFrame(corrected_table)
    st.dataframe(df, use_container_width=True)

    if updated:
        save_corrections(manual_corrections)
        save_structure(word_structure)
        st.success("Corrections sauvegardées")

    st.subheader("\U0001F4E5 Exporter vers Excel basé sur Inventaire à jour.xlsx")
    if st.button("\U0001F4E4 Télécharger l'Excel structuré"):
        if os.path.exists(TEMPLATE_FILE):
            df_template = pd.read_excel(TEMPLATE_FILE, header=None)
        else:
            df_template = pd.DataFrame()

        for entry in corrected_table:
            mot = entry["Correction manuelle"]
            row = entry["Ligne"]
            col = entry["Colonne"]
            if row and col:
                while len(df_template) < row:
                    df_template.loc[len(df_template)] = ["" for _ in range(len(df_template.columns))]
                while len(df_template.columns) < col:
                    df_template[len(df_template.columns)] = ""
                df_template.iat[row-1, col-1] = mot

        st.write("Prévisualisation du tableau structuré :")
        st.dataframe(df_template)

        output = BytesIO()
        df_template.to_excel(output, index=False, header=False, engine='openpyxl')
        today = datetime.today().strftime('%Y-%m-%d')

        projet_name = st.text_input("Nom du projet (utilisé pour le nom du fichier) :", "Inventaire à jour")
        filename = f"{projet_name}_{today}.xlsx"

        st.download_button(
            label="\U0001F4C4 Télécharger l'Excel structuré",
            data=output.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
