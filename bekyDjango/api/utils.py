from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from fuzzywuzzy import process, fuzz
import joblib
import nltk
from rasa.core.agent import Agent

model_path = '../files/20240508-113825-formal-cobbler.tar.gz'
agent = Agent.load(model_path)
predict_model = joblib.load('../files/best_model.pkl')
symptom_collection = "itching,skin_rash,nodal_skin_eruptions,continuous_sneezing,shivering,chills,joint_pain,stomach_pain,acidity,ulcers_on_tongue,muscle_wasting,vomiting,burning_micturition,spotting_urination,fatigue,weight_gain,anxiety,cold_hands_and_feets,mood_swings,weight_loss,restlessness,lethargy,patches_in_throat,irregular_sugar_level,cough,high_fever,sunken_eyes,breathlessness,sweating,dehydration,indigestion,headache,yellowish_skin,dark_urine,nausea,loss_of_appetite,pain_behind_the_eyes,back_pain,constipation,abdominal_pain,diarrhoea,mild_fever,yellow_urine,yellowing_of_eyes,acute_liver_failure,fluid_overload,swelling_of_stomach,swelled_lymph_nodes,malaise,blurred_and_distorted_vision,phlegm,throat_irritation,redness_of_eyes,sinus_pressure,runny_nose,congestion,chest_pain,weakness_in_limbs,fast_heart_rate,pain_during_bowel_movements,pain_in_anal_region,bloody_stool,irritation_in_anus,neck_pain,dizziness,cramps,bruising,obesity,swollen_legs,swollen_blood_vessels,puffy_face_and_eyes,enlarged_thyroid,brittle_nails,swollen_extremeties,excessive_hunger,extra_marital_contacts,drying_and_tingling_lips,slurred_speech,knee_pain,hip_joint_pain,muscle_weakness,stiff_neck,swelling_joints,movement_stiffness,spinning_movements,loss_of_balance,unsteadiness,weakness_of_one_body_side,loss_of_smell,bladder_discomfort,foul_smell_of_urine,continuous_feel_of_urine,passage_of_gases,internal_itching,toxic_look_(typhos),depression,irritability,muscle_pain,altered_sensorium,red_spots_over_body,belly_pain,abnormal_menstruation,dischromic_patches,watering_from_eyes,increased_appetite,polyuria,family_history,mucoid_sputum,rusty_sputum,lack_of_concentration,visual_disturbances,receiving_blood_transfusion,receiving_unsterile_injections,coma,stomach_bleeding,distention_of_abdomen,history_of_alcohol_consumption,fluid_overload,blood_in_sputum,prominent_veins_on_calf,palpitations,painful_walking,pus_filled_pimples,blackheads,scurring,skin_peeling,silver_like_dusting,small_dents_in_nails,inflammatory_nails,blister,red_sore_around_nose,yellow_crust_ooz"
symptoms_list = [symptom for symptom in symptom_collection.split(',')]

def custom_lemmatize(token, lemmatizer):
    if token.endswith("full") or token.endswith("ful"):
        return token
    else:
        return lemmatizer.lemmatize(token, get_wordnet_pos(token))

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_input(user_input):
    user_input = user_input.lower()
    tokens = word_tokenize(user_input)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [custom_lemmatize(token, lemmatizer) for token in filtered_tokens]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

def custom_fuzzy_match(query, choices, scorer=fuzz.ratio, threshold=55):
    best_match = process.extractOne(query, choices, scorer=scorer)
    if best_match and best_match[1] >= threshold:
        return best_match[0]
    return None

def extract_symptoms(user_input, symptom_keywords, threshold=55):
    tokens = word_tokenize(user_input.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    symptoms = []
    for token in lemmatized_tokens:
        matched_symptom = custom_fuzzy_match(token, symptom_keywords, fuzz.ratio, threshold)
        if matched_symptom:
            symptoms.append(matched_symptom)
    return symptoms

async def predict_intent(user_input: str) -> dict:
    processed_input = preprocess_input(user_input)
    detected_symptoms = extract_symptoms(processed_input, symptoms_list)
    if processed_input == 'hi':
        detected_symptoms = []
    if detected_symptoms:
        return {'intent': 'symptoms_additional', 'confidence': 1.0, 'symptoms': detected_symptoms}
    result = await agent.parse_message(processed_input)
    intent_name = result['intent']['name']
    intent_confidence = result['intent']['confidence']
    return {'intent': intent_name, 'confidence': intent_confidence, 'symptoms': []}

async def chat_with_bot(user_input):
    processed_input = preprocess_input(user_input)
    responses = await agent.handle_text(processed_input)
    return responses
