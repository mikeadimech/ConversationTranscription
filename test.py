from rpunct import RestorePuncts

# The default language is 'english'
rpunct = RestorePuncts(ner_args={"use_cuda": False})
print("loaded")
rpunct.punctuate("hello my name is mikea and i am twenty years old",lang="en")