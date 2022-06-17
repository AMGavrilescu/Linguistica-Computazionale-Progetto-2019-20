# encoding: UTF-8

import sys # fornisce accesso a funzioni e valori relativi all'ambiente di runtime del programma, come i parametri della riga di comando in sys.argv
import codecs # fornisce flussi e interfacce file per trascodificare dati
import re # permette il trattamento delle espressioni regolari
import nltk # permette il Language Processing e l'analisi dei testi in generale
from nltk import FreqDist # fornisce la frequenza delle parole all'interno di un testo


# operazioni preliminari

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') # caricamento modello statistico
tag_person = "PERSON" # POS tag persona 
tag_luogo = "GPE" # POS tag luogo
list_tag_sostantivi = ["NN", "NNS", "NNP", "NNPS"] # lista POS tag sostantivi
list_tag_verbi = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"] # lista POS tag verbi


# funzioni

# funzione che restituisce i token delle frasi
def ottieni_token(frasi):
    lista_token = [] # conterrà i token totali
    for frase in frasi:
        tokens = nltk.word_tokenize(frase) # ottenimento token
        lista_token += tokens # memorizzazione token
    return lista_token

# funzione che restituisce i dieci nomi propri di persona più frequenti in un testo
def ten_nomi_propri(phr):
    albero = nltk.ne_chunk(nltk.pos_tag(ottieni_token(phr))) # ottenimento token > bigrammi [token, pos] > albero
    nomi_propri = []
    for node in albero: # scorrimento nodi dell'albero
        NE = ''
        if hasattr(node, 'label'): # se nodo intermedio
            if node.label() in tag_person: # se etichetta = "PERSON"
                for part_NE in node.leaves(): # ciclo su foglie
                    NE = NE + ' ' + part_NE[0]
                nomi_propri.append(NE) # memorizzazione nome di persona trovato
    risultato = nltk.FreqDist(nomi_propri).most_common(10)
    return risultato

# funzione che restituisce bigrammi [nome, frasi in cui appare il nome]
def frasi_nomi_propri(nome, frasi):
    lista_frasi = [] # conterrà di volta in volta le frasi associate ad un nome
    lista_nome_frasi = [] # conterrà bigrammi [nome, frasi]
    for frase in frasi:
        if nome in frase: # selezione solo frasi in cui appare il nome proprio
            lista_frasi.append(frase)
    lista_nome_frasi.append([nome, lista_frasi])
    return lista_nome_frasi

# funzione che restituisce la frase più lunga in cui compare un determinato nome
def frase_lunga(nome_frasi):
    nome_frase_lunga = []
    for coppia in nome_frasi:
        lunghezza = 0 # serve per confronto
        frase_lunga = ''
        for frase in coppia[0][1]: # opera sulle frasi associate a un nome
            nr_token = len(ottieni_token(frase))
            if nr_token > lunghezza: # se il numero dei token della frase é maggiore rispetto al numero memorizzato
                lunghezza = nr_token # cambio valore lunghezza
                frase_lunga = frase # memorizzazione nuova frase lunga
        nome_frase_lunga.append([coppia[0][0], frase_lunga]) # [nome, frase lunga]
    return nome_frase_lunga

# funzione che restituisce la frase più breve in cui compare un determinato nome
def frase_breve(nome_frasi):
    nome_frase_breve = []
    for coppia in nome_frasi:
        lunghezza = 5000
        frase_breve = ''
        for frase in coppia[0][1]:
            nr_token = len(ottieni_token(frase))
            if nr_token < lunghezza: # se il numero dei token della frase é minore rispetto al numero memorizzato
                lunghezza = nr_token # cambio valore lunghezza
                frase_breve = frase # memorizzazione nuova frase breve
        nome_frase_breve.append([coppia[0][0], frase_breve]) # [nome, frase breve]
    return nome_frase_breve

# funzione che restituisce tutte le frasi in cui ci sono i nomi propri più frequenti
def solo_frasi_nome(nome_frasi):
    tot_frasi = []
    for coppia in nome_frasi:
        for frase in coppia[0][1]:
            tot_frasi.append(frase)
    return tot_frasi

# funzione che restituisce i 10 luoghi, persone, sostantivi, verbi più frequenti
def ten_lpsv(frasi):
    # liste che conterranno i risultati
    luoghi = []
    persone = []
    sostantivi = []
    verbi = []

    for frase in frasi:
        token_pos = nltk.pos_tag(nltk.word_tokenize(frase)) # ottenimento bigrammi [token, pos]
        albero = nltk.ne_chunk(token_pos) # ottenimento albero

        for node in albero:
            NE_luogo = ''
            NE_persona = ''
            if hasattr(node, 'label'): # controllo etichetta dei token
                if node.label() in tag_luogo: # se luogo
                    for part_NE in node.leaves():
                        NE_luogo = NE_luogo + ' ' + part_NE[0]
                    luoghi.append(NE_luogo) # memorizzazione token in lista luoghi

                if node.label() in tag_person: # se persona
                    for part_NE in node.leaves(): 
                        NE_persona = NE_persona + ' ' + part_NE[0]
                    persone.append(NE_persona) # memorizzazione token in lista persone

        for tok, pos in token_pos:
            if pos in list_tag_sostantivi: # se il token é etichettato come sostantivo
                sostantivi.append(tok) # memorizzazione token in lista sostantivi

            if pos in list_tag_verbi: # se il token é etichettato come verbo
                verbi.append(tok) # memorizzazione token in lista verbi

    # ordinamento per frequenza e selezione dei primi 10 risultati
    ten_luoghi = nltk.FreqDist(luoghi).most_common(10)
    ten_persone = nltk.FreqDist(persone).most_common(10)
    ten_sostantivi = nltk.FreqDist(sostantivi).most_common(10)
    ten_verbi = nltk.FreqDist(verbi).most_common(10)
    return ten_luoghi, ten_persone, ten_sostantivi, ten_verbi

# funzione che restituisce date, mesi e giorni della settimana estratti attraverso l'utilizzo di espressioni regolari
def date_mesi_giorni(frasi):
    date = []
    mesi = []
    giorni = []
    for frase in frasi:
        # regex per match date:
        # dd-mm-yyyy
        date += re.findall(r'\b([1-9]|0[1-9]|1\d|2\d|3[0-1])[.\-\/\\]([1-9]|0\d|1[0-2])[.\-\/\\](\d\d|1\d\d\d|19\d\d|20\d\d)\b', frase)
        # mm-dd-yyyy
        date += re.findall(r'\b([1-9]|0\d|1[0-2])[.\-\/\\]([1-9]|0[1-9]|1\d|2\d|3[0-1])[.\-\/\\](\d\d|1\d\d\d|19\d\d|20\d\d)\b', frase) 
        # yyyy-mm-dd
        date += re.findall(r'\b(\d\d|1\d\d\d|19\d\d|20\d\d)[.\-\/\\]([1-9]|0\d|1[0-2])[.\-\/\\]([1-9]|0[1-9]|1\d|2\d|3[0-1])\b', frase) 
        # regex per match mesi
        mesi += re.findall(r'\b(?:Jan|Febr)uary|March|April|May|Ju(?:ne|ly)|August|October|(?:Sept|Nov|Dec)ember\b', frase) 
        # regex per match giorni
        giorni += re.findall(r'\b(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\b', frase) 
    date = nltk.FreqDist(date).most_common()
    mesi = nltk.FreqDist(mesi).most_common()
    giorni = nltk.FreqDist(giorni).most_common()
    return date, mesi, giorni

# funzione che restituisce la frase con probabilità più alta tra tutte le frasi associate ai nomi propri
def frase_markov(dist_freq, len_corpus, frasi_nome):
    risultato = []
    max_prob = 0
    max_frase = ''
    for frase in frasi_nome:
        toks = nltk.word_tokenize(frase)
        if len(toks) > 7 and len(toks) < 13: # la frase dev'essere lunga da 8 a 12 token
            prob_markov = 1.0
            for tok in toks:
                probabilita = dist_freq[tok]*1.0 / len_corpus*1.0
                prob_markov = prob_markov * probabilita # probabilità modello markov di ordine 0: P(A1,...,An) = P(A1)*P(A2)*...*P(An)
            if prob_markov > max_prob: # viene sostituito il valore memorizzato precedentemente con quello maggiore, se trovato
                max_prob = prob_markov
                max_frase = frase # memorizzazione frase con probabilità più alta
    risultato.append([max_frase, max_prob])
    return risultato


# funzione principale

def main(file1, file2):

    print("Progetto di Linguistica Computazionale, a.a. 2019/2020\n*** PROGRAMMA 2 ***\n\nAutrice: Ana Maria Gavrilescu\nMatricola: 578495\n\nTesti:\n- Crime and Punishment by Fyodor Dostoyevsky\n- Mansfield Park by Jane Austen\n\n\n")

    # apertura file passati come input al programma 
    open_1 = codecs.open(file1, "r", "utf-8")
    open_2 = codecs.open(file2, "r", "utf-8")

    # lettura file 
    read_1 = open_1.read()
    read_2 = open_2.read()

    # estrazione frasi
    phr_1 = tokenizer.tokenize(read_1)
    phr_2 = tokenizer.tokenize(read_2)


    # inizio richieste progetto

    print("1) Estrazione dieci nomi propri di persona più frequenti e relative liste delle frasi che li contengono\n")

    ten_nomi_propri_1 = ten_nomi_propri(phr_1) # calcolo lista [nome proprio, frequenza] richiamando funzione
    ten_nomi_propri_2 = ten_nomi_propri(phr_2)

    nome_frasi_1 = []
    for nome in ten_nomi_propri_1:
        coppia = frasi_nomi_propri(nome[0], phr_1) # calcolo associazione nome, frasi in cui é contenuto richiamando funzione
        nome_frasi_1.append(coppia)
    
    nome_frasi_2 = []
    for nome in ten_nomi_propri_2:
        coppia = frasi_nomi_propri(nome[0], phr_2)
        nome_frasi_2.append(coppia)

    # primo testo
    print("Nel primo testo, i dieci nomi propri di persona più frequenti sono:\n")
    for i in ten_nomi_propri_1:
        print(str(i[0]))

    print("\n")

    print("Lista delle frasi in cui appaiono i nomi:\n")
    for i in nome_frasi_1:
        print("Il nome '" + str(i[0][0]) + " ' appare in queste frasi:")
        print (str(i[0][1]) + "\n")
    
    # secondo testo
    print("Nel secondo testo, i dieci nomi propri di persona più frequenti sono:\n")
    for i in ten_nomi_propri_2:
        print(str(i[0]))

    print("\n")

    print("Lista delle frasi in cui appaiono i nomi:\n")
    for i in nome_frasi_2:
        print("Il nome '" + str(i[0][0]) + " ' appare in queste frasi:")
        print (str(i[0][1]) + "\n")


    print("2) Frase più lunga e frase più breve che contengono i nomi trovati\n")
    
    # primo testo
    print("Per i nomi propri più frequenti nel primo testo:\n")

    print("Frasi più lunghe:\n")
    frase_lunga_1 = frase_lunga(nome_frasi_1) # calcolo frase più lunga per ogni nome richiamando funzione
    for elem in frase_lunga_1:
        print("• La frase più lunga per il nome '" + str(elem[0]) + " ' é:\n" + str(elem[1]) + "\n\n")

    print("Frasi più brevi:\n")
    frase_breve_1 = frase_breve(nome_frasi_1) #calcolo frase più breve richiamando funzione
    for elem in frase_breve_1:
        print("• La frase più breve per il nome '" + str(elem[0]) + " ' é:\n" + str(elem[1]) + "\n")
    
    print("\n")
    
    # secondo testo
    print("Per i nomi propri più frequenti nel secondo testo:\n")

    print("Frasi più lunghe:\n")
    frase_lunga_2 = frase_lunga(nome_frasi_2)
    for elem in frase_lunga_2:
        print("• La frase più lunga per il nome '" + str(elem[0]) + " ' é:\n" + str(elem[1]) + "\n\n")

    print("Frasi più brevi:\n")
    frase_breve_2 = frase_breve(nome_frasi_2)
    for elem in frase_breve_2:
        print("• La frase più breve per il nome '" + str(elem[0]) + " ' é:\n" + str(elem[1]) + "\n")
        
    print("\n")
    
    print("3) Analizzando solo le frasi dove compare il nome proprio, estrarre ed ordinare in ordine di frequenza decrescente, indicando anche la relativa frequenza:\n")

    # ottenimento totale frasi in cui appaiono i nomi propri
    solo_frasi_nome_1 = solo_frasi_nome(nome_frasi_1)
    solo_frasi_nome_2 = solo_frasi_nome(nome_frasi_2)


    # calcolo 10 luoghi, persone, sostantivi, verbi più frequenti richiamando funzione
    ten_luoghi_1, ten_persone_1, ten_sostantivi_1, ten_verbi_1 = ten_lpsv(solo_frasi_nome_1)
    ten_luoghi_2, ten_persone_2, ten_sostantivi_2, ten_verbi_2 = ten_lpsv(solo_frasi_nome_2)        
    
    print("I 10 luoghi, persone, sostantivi, verbi più frequenti:\n\n")

    # primo testo
    print("Nel primo testo:\n")
    for coppia in ten_luoghi_1:
        print("Il luogo '" + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    print("\n")
    for coppia in ten_persone_1:
        print("La persona '" + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    print("\n")
    for coppia in ten_sostantivi_1:
        print("Il sostantivo ' " + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    print("\n")
    for coppia in ten_verbi_1:
        print("Il verbo ' " + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    
    print("\n")

    # secondo testo
    print("Nel secondo testo:\n")
    for coppia in ten_luoghi_2:
        print("Il luogo '" + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    print("\n")
    for coppia in ten_persone_2:
        print("La persona '" + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    print("\n")
    for coppia in ten_sostantivi_2:
        print("Il sostantivo ' " + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    print("\n")
    for coppia in ten_verbi_2:
        print("Il verbo ' " + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    
    print("\n")  

    print("Le date, i mesi e i giorni della settimana estratti attraverso l'utilizzo di espressioni regolari\n")

    # calcolo date, mesi, giorni richiamando funzione
    date_1, mesi_1, giorni_1 = date_mesi_giorni(solo_frasi_nome_1)    
    date_2, mesi_2, giorni_2 = date_mesi_giorni(solo_frasi_nome_2)    

    # primo testo
    print("Nel primo testo:\n")
    if date_1 == []:
        print("Non sono state trovate date")
    if mesi_1 == []:
        print("Non sono stati trovati mesi")
    if giorni_1 == []:
        print("Non sono stati trovati giorni della settimana")

    for coppia in date_1:
            print("La data ' " + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    for coppia in mesi_1:
            print("Il mese ' " + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    for coppia in giorni_1:
            print("Il giorno della settimana ' " + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    
    print("\n")

    # secondo testo
    print("Nel secondo testo:\n")
    if date_2 == []:
        print("Non sono state trovate date")
    if mesi_2 == []:
        print("Non sono stati trovati mesi")
    if giorni_2 == []:
        print("Non sono stati trovati giorni della settimana")

    for coppia in date_2:
            print("La data ' " + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    for coppia in mesi_2:
            print("Il mese ' " + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    for coppia in giorni_2:
            print("Il giorno della settimana ' " + str(coppia[0]) + " ' appare " + str(coppia[1]) + " volta/e")
    
    print("\n")

    print("La frase lunga minimo 8 token e massimo 12 con probabilità più alta (calcolata attraverso un modello di Markov di ordine 0 che usa le distribuzioni di frequenza estratte dall’intero corpus)\n")
    
    # token dei file 
    tok_1 = nltk.word_tokenize(read_1)
    tok_2 = nltk.word_tokenize(read_2)

    # lunghezza corpus
    len_1 = len(tok_1)
    len_2 = len(tok_2)

    # distribuzione di frequenza intero corpus
    dist_freq_txt_1 = nltk.FreqDist(tok_1)
    dist_freq_txt_2 = nltk.FreqDist(tok_2)

    # calcolo, usando la distribuzione di frequenza dell'intero corpus, la frase (tra quelle in cui appaiono i nomi propri) con probabilità più alta, usando un modello markov di ordine 0 richiamando la funzione; la cardinalità del corpus serve per calcolare la probabilità dei singoli token: frequenza del token/numero totale token
    frase_markov_1 = frase_markov(dist_freq_txt_1, len_1, solo_frasi_nome_1)
    frase_markov_2 = frase_markov(dist_freq_txt_2, len_2, solo_frasi_nome_2)

    for elem in frase_markov_1:
        print("Nel primo testo é:\n' " + str(elem[0]) + " ' con probabilità " + str(elem[1]) + "\n")
    for elem in frase_markov_2:
        print("Nel secondo testo é:\n' " + str(elem[0]) + " ' con probabilità " + str(elem[1]))

# richiamo funzione principale passandole in input i file scelti
main(sys.argv[1], sys.argv[2])

# Una volta nel path della directory del progetto nel terminale, per avviare il programma, digitare " python3 programma2.py testo1.txt testo2.txt "
# Per vidualizzare il risultato in un file di testo che verrà visulizzato nella directory del progetto, aggiungere " > output2.txt "