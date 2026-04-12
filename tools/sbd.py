import re


def isTermPunct(token):
    return token in {".", "?", "!"}


def isWordCap(word):
    return word[0].isupper() if word else False


def is_abbr1(word):
    ABBREVIATIONS = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
        "st", "vs", "etc", "u.s", "u.k"
    }
    return word.lower() in ABBREVIATIONS




def splitParaList(text):
   
    tokens = re.findall(r"\w+|[.!?]", text)
    
    chunks = []
  
    for i in range(1, len(tokens) - 1):
        if isTermPunct(tokens[i]):
            chunks.append([tokens[i-1], tokens[i], tokens[i+1]])
  
    if len(tokens) >= 2 and isTermPunct(tokens[-1]):
        chunks.append([tokens[-2], tokens[-1], "$$"])
    
    return chunks


def checkRules(triple):
    left, punct, right = triple
    
    if not isTermPunct(punct):
        return False
    if is_abbr1(left):
        return False
   
    if left.isdigit() and right.isdigit():
        return False
    if right == "$$" or isWordCap(right):
        return True
    
    return False


def findBoundary(strlist):
    valid_chunks = [chunk for chunk in strlist if checkRules(chunk)]
    boundary = [chunk[0] for chunk in valid_chunks]
    return boundary



def buildSentences(text, boundary):
    words = text.split(" ")
    boundary_set = set(boundary)
    
    sentences = []
    current = []
    
    for word in words:
        current.append(word)
        
        clean_word = word.strip(".!?")
        
        if clean_word in boundary_set:
            sentences.append(" ".join(current))
            current = []
    
    if current:
        sentences.append(" ".join(current))
    
    return sentences


def sentencePipeline(text):
    chunks = splitParaList(text)
    boundary = findBoundary(chunks)
    sentences = buildSentences(text, boundary)
    
    return sentences

