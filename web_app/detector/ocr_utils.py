"""
OCR utility for extracting name from ID card images using EasyOCR.
"""
import re
import logging

logger = logging.getLogger(__name__)


def extract_name_from_id(image_path):
    """
    Extract first name and last name from an ID card image.
    Returns (first_name, last_name) tuple.
    """
    try:
        import easyocr
        reader = easyocr.Reader(['en', 'fr'], gpu=False, verbose=False)
        results = reader.readtext(str(image_path))
        
        # Collect all detected text, keeping them ordered top-to-bottom
        all_text = [r[1] for r in results]
        logger.info(f"OCR detected text: {all_text}")
        
        first_name = ""
        last_name = ""
        
        # We will parse the text looking for explicit markers like "NOM:", "PRÉNOM:"
        # OCR might return ['NOM: ZAABOUTI', 'PRÉNOM : Ilyess'] or ['NOM', ':', 'ZAABOUTI']
        
        # Normalize the whole text for regex
        combined_text = " ".join(all_text)
        
        # Very forgiving regex: Look for NOM or PRENOM, allow any non-letters between it and the name
        nom_match = re.search(r'(?i)(?:NOM|NAME|SURNAME)[^a-zA-ZÀ-ÿ]*([A-Za-zÀ-ÿ]{2,})', combined_text)
        prenom_match = re.search(r'(?i)(?:PR[EÉ]NOM|PRENOM|FIRST NAME)[^a-zA-ZÀ-ÿ]*([A-Za-zÀ-ÿ]{2,})', combined_text)
        
        if nom_match:
            last_name = nom_match.group(1).strip().title()
        if prenom_match:
            first_name = prenom_match.group(1).strip().title()
            
        # Fallback: token-by-token scan
        if not last_name or not first_name:
            for i, text in enumerate(all_text):
                clean = text.strip().upper()
                
                # If we see NOM and don't have last_name yet
                if 'NOM' in clean and not 'PRENOM' in clean and not 'PRÉNOM' in clean and not last_name:
                    # The name might be in the same string: "NOM: ZAABOUTI"
                    parts = re.split(r'[:\-]', text)
                    if len(parts) > 1 and len(parts[-1].strip()) > 1:
                        last_name = parts[-1].strip().title()
                    # Or it's in the next array element
                    elif i + 1 < len(all_text):
                        # Skip colons if OCR put them in their own element
                        next_idx = i + 1
                        if all_text[next_idx].strip() == ':' and next_idx + 1 < len(all_text):
                            next_idx += 1
                        last_name = all_text[next_idx].strip().title()
                        
                # If we see PRENOM and don't have first_name yet
                if ('PRENOM' in clean or 'PRÉNOM' in clean) and not first_name:
                    parts = re.split(r'[:\-]', text)
                    if len(parts) > 1 and len(parts[-1].strip()) > 1:
                        first_name = parts[-1].strip().title()
                    elif i + 1 < len(all_text):
                        next_idx = i + 1
                        if all_text[next_idx].strip() == ':' and next_idx + 1 < len(all_text):
                            next_idx += 1
                        first_name = all_text[next_idx].strip().title()
                    
        # 3. Last resort fallback if we still don't have them
        if not first_name and not last_name:
            name_candidates = []
            skip_words = {'carte', 'etudiant', 'étudiant', 'esprit', 'honoris', 'united', 'universities', 'nom', 'prenom', 'prénom', 'classe', 'identifiant', 'annee', 'année', 'universitaire'}
            for text in all_text:
                clean = text.strip()
                if len(clean) < 2 or re.search(r'\d', clean):
                    continue
                if clean.lower() in skip_words:
                    continue
                if re.match(r'^[A-Za-zÀ-ÿ\s\-]+$', clean):
                    name_candidates.append(clean)
            if len(name_candidates) >= 2:
                last_name = name_candidates[0].strip().title()
                first_name = name_candidates[1].strip().title()
            elif len(name_candidates) == 1:
                first_name = name_candidates[0].strip().title()
                last_name = "User"
                
        # Final cleanup
        if not first_name: first_name = "Unknown"
        if not last_name: last_name = "User"
        
        first_name = re.sub(r'[^A-Za-zÀ-ÿ]', '', first_name)
        last_name = re.sub(r'[^A-Za-zÀ-ÿ]', '', last_name)
        
        return first_name, last_name
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return "Unknown", "User"


def generate_email(first_name, last_name):
    """Generate a corporate email from first and last name."""
    first = re.sub(r'[^a-zA-Z]', '', first_name).lower()
    last = re.sub(r'[^a-zA-Z]', '', last_name).lower()
    if first and last:
        return f"{first}_{last}@EcoGuard.ai"
    elif first:
        return f"{first}@EcoGuard.ai"
    return "user@EcoGuard.ai"
