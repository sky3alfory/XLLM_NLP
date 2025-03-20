# pip install spacy==3.5.0 
# pip install spacy-transformers==1.1.5
# pip install transformers==4.30.0
# pip install huggingface-hub==0.16.4
# python -m spacy download en_core_web_trf

#### cuda11.7 과 연동이 안되는듯
# pip install torch==1.13.1+cu114 --index-url https://download.pytorch.org/whl/cu113
# pip install -U transformers
# pip install -U spacy
# python -m spacy download en_core_web_trf


#### 교재 가이드라인
pip install -U spacy[cuda110,transformers,lookups]==3.0.3
pip install -U spacy-lookups-data==1.0.0
pip install cupy-cuda110==8.5.0
python -m spacy download en_core_web_trf