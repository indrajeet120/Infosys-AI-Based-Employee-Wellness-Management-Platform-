import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from services.config import DATA_DIR, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, ISEAR_BENCHMARK_PATH

# Training samples with single and multi-label emotions
TRAIN_RECORDS = [
    # Joy
    ("I am overjoyed by the wonderful promotion and recognition at work!", "joy"),
    ("We celebrated our anniversary with so much laughter and happiness.", "joy"),
    ("Winning first place in the championship was the happiest moment of my life.", "joy"),
    ("I absolutely love spending time with my family on sunny weekends.", "joy"),
    ("My dream finally came true today and I feel completely ecstatic!", "joy"),
    ("The concert was spectacular and everyone was dancing with delight.", "joy"),
    ("I received the best birthday present ever from my closest friends.", "joy"),
    ("What a fantastic and uplifting performance by the orchestra!", "joy"),
    ("I am so proud of your hard work and immense success!", "joy"),
    ("Everything is going smoothly and my heart is filled with gratitude.", "joy"),
    
    # Sadness
    ("I feel deeply depressed and heartbroken after hearing the tragic news.", "sadness"),
    ("Losing my faithful dog of twelve years has left me completely shattered.", "sadness"),
    ("I tried so hard for this examination but failed miserably.", "sadness"),
    ("The loneliness in this empty house is overwhelming and depressing.", "sadness"),
    ("It hurts so much to see someone you cherish walk out of your life.", "sadness"),
    ("My application was rejected and all my hopes are crushed.", "sadness"),
    ("I cannot stop crying over the painful memories from my past.", "sadness"),
    ("A feeling of profound grief and sorrow enveloped the entire family.", "sadness"),
    ("The gloomy weather matches the heavy despair inside my chest.", "sadness"),
    ("I feel completely abandoned and forgotten by everyone I trusted.", "sadness"),

    # Anger
    ("I am absolutely furious at the corrupt politicians ruining our city!", "anger"),
    ("Their blatant lies and disrespect made my blood boil with rage.", "anger"),
    ("How dare they insult my team after all the effort we put in!", "anger"),
    ("I am extremely pissed off by this terrible customer service.", "anger"),
    ("The careless driver almost ran over a pedestrian and showed zero remorse.", "anger"),
    ("Stop yelling at me and treat people with basic decency!", "anger"),
    ("It infuriates me when colleagues steal credit for my contributions.", "anger"),
    ("The landlord cheated us out of our security deposit and refused to talk.", "anger"),
    ("I am outraged by the unfair treatment and biased rules in this competition.", "anger"),
    ("This delayed flight and rude airline staff drove me mad with anger.", "anger"),

    # Fear
    ("I was terrified when the earthquake shook the building violently.", "fear"),
    ("The dark alley at midnight gave me chills and intense anxiety.", "fear"),
    ("I am panicking about the upcoming surgery and its potential complications.", "fear"),
    ("Hearing loud unexplained footsteps outside my window scared me to death.", "fear"),
    ("I have an overwhelming dread of public speaking and stage fright.", "fear"),
    ("The turbulent flight dropped suddenly, causing pure horror among passengers.", "fear"),
    ("I tremble with fear whenever I remember that near-fatal car accident.", "fear"),
    ("The doctor's serious expression made me worry about the worst diagnosis.", "fear"),
    ("She felt a paralyzing dread creeping in as the stranger approached.", "fear"),
    ("The thought of losing my livelihood keeps me awake with anxiety all night.", "fear"),

    # Surprise
    ("I was totally stunned when they threw a surprise party for me!", "surprise"),
    ("Never in a million years did I expect to bump into my childhood teacher here.", "surprise"),
    ("The sudden plot twist at the climax left the entire theater speechless.", "surprise"),
    ("Wow, look at those astonishing northern lights illuminating the sky!", "surprise"),
    ("I was caught completely off guard by the unexpected announcement.", "surprise"),
    ("It was an incredible shock to discover our team actually won the lottery!", "surprise"),
    ("My jaw dropped when I saw the magical transformation of the old castle.", "surprise"),
    ("I couldn't believe my eyes when I saw the celebrity standing right next to me.", "surprise"),
    ("The sudden fireworks explosion took everyone by surprise.", "surprise"),
    ("What an unexpected and curious turn of events this morning!", "surprise"),

    # Disgust
    ("The rotten smell of spoiled food in the refrigerator was utterly repulsive.", "disgust"),
    ("I felt nauseated seeing the filthy and unhygienic conditions of the kitchen.", "disgust"),
    ("His cruel and selfish behavior toward animals disgusts me to the core.", "disgust"),
    ("There were slimy worms crawling all over the spoiled vegetables, so gross!", "disgust"),
    ("I cannot tolerate such vulgar language and abhorrent attitudes.", "disgust"),
    ("The foul stench of the open sewer made me want to throw up immediately.", "disgust"),
    ("His slimy tactics and deceitful lies fill me with pure revulsion.", "disgust"),
    ("The moldy bread was covered in green fungus, utterly sickening.", "disgust"),
    ("I was repulsed by the offensive and slimy remarks made during the meeting.", "disgust"),
    ("The greasy, contaminated water in the pond was truly revolting.", "disgust"),

    # Multi-label combinations
    ("I am thrilled about moving abroad for my new job, but terrified of living alone.", "joy, fear"),
    ("She felt a rush of joyful excitement mixed with anxious nervousness before her wedding.", "joy, fear"),
    ("I was pleasantly surprised and overjoyed by the unexpected award ceremony.", "joy, surprise"),
    ("The surprise gift brought tears of happiness and pure delight to her eyes.", "joy, surprise"),
    ("I am heartbroken and furiously angry at the betrayal by my business partner.", "sadness, anger"),
    ("The sudden death of our mentor filled us with deep grief and shock.", "sadness, surprise"),
    ("Seeing the filthy litter strewn across the sacred temple made me disgusted and angry.", "anger, disgust"),
    ("I was revolted by his cruel lies and raged at his arrogance.", "anger, disgust"),
    ("The creepy, slime-covered monster in the horror movie was both terrifying and disgusting.", "fear, disgust"),
    ("The sudden appearance of the venomous snake gave me a huge shock and panic.", "fear, surprise"),
    ("I was astonished and deeply saddened to learn about the sudden closure of the school.", "sadness, surprise"),
    ("He felt both enraged by the insult and frightened by the violent threats.", "anger, fear"),
]

VAL_RECORDS = [
    ("I am so grateful for all the love and support from my wonderful family.", "joy"),
    ("My heart sank when I saw the disappointing test results on the screen.", "sadness"),
    ("It makes me so angry when drivers do not use their turn signals.", "anger"),
    ("I get nervous and scared when walking home alone in the pitch dark.", "fear"),
    ("I had no idea you were coming to visit today, what a pleasant shock!", "surprise"),
    ("The dirty public restroom was smelling awful and completely nauseating.", "disgust"),
    ("I am excited to start university but dreading the difficult exams.", "joy, fear"),
    ("Finding out about the fraud was shocking and deeply upsetting for all of us.", "sadness, surprise"),
    ("His abusive behavior was both revolting and infuriating.", "anger, disgust"),
    ("The unexpected storm terrified everyone with its violent lightning.", "fear, surprise"),
]

TEST_RECORDS = [
    ("I received my dream job offer today and cannot stop smiling!", "joy"),
    ("The sorrow of parting with my best friend is too heavy to bear.", "sadness"),
    ("I am furious that my order was cancelled without any explanation!", "anger"),
    ("The loud bang in the middle of the night made me jump in sheer panic.", "fear"),
    ("What a surprise to see snow falling in the desert this morning!", "surprise"),
    ("The sight of moldy decayed fruit was totally nauseating and gross.", "disgust"),
    ("I am happy about winning the race but scared about defending the title.", "joy, fear"),
    ("I was shocked and overjoyed when my book became a bestseller!", "joy, surprise"),
    ("Their deceitful conduct made me feel sick with anger and disgust.", "anger, disgust"),
    ("The sudden loss of our family home in the fire left us horrified and grieving.", "sadness, fear"),
]

# Held-out ISEAR benchmark subset (Independent from training data)
ISEAR_RECORDS = [
    # ISEAR Joy
    ("When I passed my final university examinations with high honors.", "joy"),
    ("When my partner proposed to me during our vacation in Paris.", "joy"),
    ("When I was reunited with my family after three years of separation.", "joy"),
    ("When our team won the annual regional football tournament.", "joy"),
    
    # ISEAR Sadness
    ("When my grandfather passed away after a prolonged illness.", "sadness"),
    ("When I realized that my long-term relationship was definitively over.", "sadness"),
    ("When I was diagnosed with a chronic health condition.", "sadness"),
    ("When I had to say farewell to all my childhood friends before emigrating.", "sadness"),
    
    # ISEAR Anger
    ("When someone falsely accused me of stealing money from the office.", "anger"),
    ("When a colleague took total credit for the project I completed single-handedly.", "anger"),
    ("When my bicycle was stolen from outside my apartment building.", "anger"),
    ("When an aggressive customer insulted my staff without any justification.", "anger"),
    
    # ISEAR Fear
    ("When I was driving on an icy highway and the vehicle suddenly spun out of control.", "fear"),
    ("When I had to undergo an emergency medical operation under general anesthesia.", "fear"),
    ("When I got lost in a dense forest as darkness fell rapidly.", "fear"),
    ("When an armed robbery occurred at the convenience store while I was shopping.", "fear"),
    
    # ISEAR Disgust
    ("When I found a dead insect inside my soup at an expensive restaurant.", "disgust"),
    ("When I witnessed someone spit directly on a public bus seat.", "disgust"),
    ("When I smelled the overwhelming odor of rotting garbage on a hot summer day.", "disgust"),
    ("When I saw an individual mistreat a defenseless stray animal on the street.", "disgust"),
    
    # ISEAR Surprise
    ("When I opened the front door to find all my distant relatives standing there.", "surprise"),
    ("When an unexpected inheritance from a distant great-uncle was announced.", "surprise"),
    ("When the weather suddenly changed from clear blue sky to an abrupt thunderstorm.", "surprise"),
]


def create_and_save_datasets():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    df_train = pd.DataFrame(TRAIN_RECORDS, columns=["text", "emotions"])
    df_val = pd.DataFrame(VAL_RECORDS, columns=["text", "emotions"])
    df_test = pd.DataFrame(TEST_RECORDS, columns=["text", "emotions"])
    df_isear = pd.DataFrame(ISEAR_RECORDS, columns=["text", "emotions"])
    
    df_train.to_csv(TRAIN_DATA_PATH, index=False)
    df_val.to_csv(VAL_DATA_PATH, index=False)
    df_test.to_csv(TEST_DATA_PATH, index=False)
    df_isear.to_csv(ISEAR_BENCHMARK_PATH, index=False)
    
    print(f"Created Train dataset: {len(df_train)} samples -> {TRAIN_DATA_PATH}")
    print(f"Created Val dataset: {len(df_val)} samples -> {VAL_DATA_PATH}")
    print(f"Created Test dataset: {len(df_test)} samples -> {TEST_DATA_PATH}")
    print(f"Created ISEAR benchmark dataset: {len(df_isear)} samples -> {ISEAR_BENCHMARK_PATH}")


if __name__ == "__main__":
    create_and_save_datasets()
