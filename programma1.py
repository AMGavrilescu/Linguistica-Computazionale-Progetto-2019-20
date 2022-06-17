# encoding: UTF-8

import sys # fornisce accesso a funzioni e valori relativi all'ambiente di runtime del programma, come i parametri della riga di comando in sys.argv
import codecs # fornisce flussi e interfacce file per trascodificare dati
import math # permette operazioni matematiche
import nltk # permette il Language Processing e l'analisi dei testi in generale
from nltk import bigrams # permette di ottenere i bigrammi di un testo
from nltk import FreqDist # fornisce la frequenza delle parole all'interno di un testo
from collections import Counter # conta l'occorrenza di un elemento


# operazioni preliminari

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') # caricamento modello statistico
lista_tag_sostantivi = ["NN", "NNS", "NNP", "NNPS"] # lista POS tag sostantivi
lista_tag_verbi = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"] # lista POS tag verbi


# funzioni

# funzione che restituisce la media delle parole in termini di caratteri
def len_med_word(lista_tok, nr_tok):
    tot_car = 0 # conterrà il numero totale dei caratteri
    for tok in lista_tok:
        tot_car += len(tok) # viene calcolato il numero di caratteri del token e sommato al numero precedente
    risultato = tot_car*1.0 / nr_tok*1.0 # divido il numero dei caratteri per il numero dei token
    return risultato

# funzione che restituisce la distribuzione delle parole tipo all'aumentare del corpus per porzioni incrementali
def distr_voc(tok_list, index):
    temp_list = [] # conterrà i token 
    for tok in range(0, index):
        if tok < index: # controllo non superamento quantità token
            temp_list.append(tok_list[tok]) # memorizzazione token
    vocabolario = set(temp_list) # la funzione set permette di ottenere la lista degli elementi diversi
    return len(vocabolario) # dimensione vocabolario

# funzione che restituisce la distribuzione degli hapax all'aumentare del corpus per porzioni incrementali
def distr_hapax(tok_list, index):
    temp_list = []
    nr_hapax = 0
    for tok in range(0, index):
        if tok < index:
            temp_list.append(tok_list[tok])
    dist_freq = Counter(temp_list).items() # calcolo la frequenza dei token
    for i in dist_freq:
        if i[1] ==  1: # se frequenza = 1, cioè é un hapax 
            nr_hapax += 1 # incremento contatore degli hapax
    return nr_hapax

# funzione che restituisce il rapporto tra sostantivi e verbi nel testo
def rapporto_sv(tok_annotati):
    sostantivi = [] # conterrà i token etichettati come sostantivi
    verbi = [] # conterrà i token etichettati come verbi
    for token, pos in tok_annotati:
        if pos in lista_tag_sostantivi:
            sostantivi.append(token)
        if pos in lista_tag_verbi:
            verbi.append(token)
    nr_sost = len(sostantivi) # numero token etichettati come sostantivi
    nr_verb = len(verbi) # numero token etichettati come verbi
    risultato = nr_sost*1.0 / nr_verb*1.0
    return risultato

# funzione che restituisce soltanto le PoS del testo
def solo_PoS(tok_annotati):
    list_pos = []
    for elem in tok_annotati:
        list_pos.append(elem[1]) # in posizione 1 ci sarà la PoS
    return list_pos

# funzione che restituisce i 10 bigrammi PoS con probabilità condizionata e forza associativa massime
def ten_bgr_pos_max(pos, bigr_pos):
    risultato_prob_condizionata = [] # conterrà i 10 bigrammi PoS con probabilità condizionata massima
    risultato_forza_associativa = [] # conterrà i 10 bigrammi PoS con forza associativa massima
    n = len(pos)
    distr_freq = nltk.FreqDist(bigr_pos)
    
    for bigramma in distr_freq:
        freq_I_elem = pos.count(bigramma[0]) # calcolo frequenza primo elemento
        freq_bigr = distr_freq[bigramma] # calcolo frequenza bigramma
        prob_cond = freq_bigr*1.0 / freq_I_elem*1.0 # calcolo probabilità condizionata (frequenza bigramma / frequenza primo elemento)
        risultato_prob_condizionata.append([prob_cond, bigramma]) 
        
        freq_b = pos.count(bigramma[1]) # calcolo frequenza secondo elemento
        prodotto_bigr_n = (freq_bigr) * n # f(<u,v>)*n
        prodotto_a_b = freq_I_elem * freq_b # f(u)*f(v)
        log = math.log(prodotto_bigr_n / prodotto_a_b, 2) # log2(f(<u,v>)*n / f(u)*f(v))
        lmi = freq_bigr * log # LMI = f(<u,v>) * MI; MI = log2(O/E) = log2(f(<u,v>)*n / f(u)*f(v))
        risultato_forza_associativa.append([lmi, bigramma])
    
    risultato_prob_condizionata = sorted(risultato_prob_condizionata, reverse = True) # ordinamento in ordine decrescente
    risultato_forza_associativa = sorted(risultato_forza_associativa, reverse = True) # ordinamento in ordine decrescente
    return risultato_prob_condizionata[:10], risultato_forza_associativa[:10] # restituzione primi 10 risultati probabilità massime


# funzione principale

def main(file1, file2):

    print("Progetto di Linguistica Computazionale, a.a. 2019/2020\n*** PROGRAMMA 1 ***\n\nAutrice: Ana Maria Gavrilescu\nMatricola: 578495\n\nTesti:\n- Crime and Punishment by Fyodor Dostoyevsky\n- Mansfield Park by Jane Austen\n\n\n")

    # apertura file passati come input al programma 
    open_1 = codecs.open(file1, "r", "utf-8")
    open_2 = codecs.open(file2, "r", "utf-8")

    # lettura file 
    read_1 = open_1.read()
    read_2 = open_2.read()

    # estrazione frasi
    phr_1 = tokenizer.tokenize(read_1)
    phr_2 = tokenizer.tokenize(read_2)

    # estrazione token
    tok_1 = nltk.word_tokenize(read_1)
    tok_2 = nltk.word_tokenize(read_2)


    # inizio richieste progetto

    print("1) Numero totale di frasi e di token\n")

    # numero totale di frasi
    nr_frasi_1 = len(phr_1)
    nr_frasi_2 = len(phr_2)

    print("Il numero totale delle frasi del primo testo é: " + str(nr_frasi_1))
    print("Il numero totale delle frasi del secondo testo é: " + str(nr_frasi_2))
    
    # comparazione testi in base al numero di frasi
    if nr_frasi_1 > nr_frasi_2:
        print("Il primo testo ha un maggior numero di frasi rispetto al secondo testo\n")
    elif nr_frasi_1 < nr_frasi_2:
        print("Il secondo testo ha un maggior numero di frasi rispetto al primo testo\n")
    else:
        print("I due testi hanno lo stesso numero di frasi\n")
    
    # numero totale di token
    nr_tok_1 = len(tok_1)
    nr_tok_2 = len(tok_2)

    print("Il numero totale dei token del primo testo é: " + str(nr_tok_1))
    print("Il numero totale dei token del secondo testo é: " + str(nr_tok_2))

    # comparazione testi in base al numero di token
    if nr_tok_1 > nr_tok_2:
        print("Il primo testo ha un maggior numero di token rispetto al secondo testo\n\n")
    elif nr_tok_1 < nr_tok_2:
        print("Il secondo testo ha un maggior numero di token rispetto al primo testo\n\n")
    else:
        print("I due testi hanno lo stesso numero di token\n\n")


    print("2) Lunghezza media delle frasi in termini di token e lunghezza media delle parole in termini di caratteri\n")

    # calcolo lunghezza media delle frasi in termini di token
    len_med_phr_1 = nr_tok_1*1.0/nr_frasi_1*1.0
    len_med_phr_2 = nr_tok_2*1.0/nr_frasi_2*1.0

    print("La lunghezza media delle frasi in termini di token del primo testo é: " + str(len_med_phr_1))
    print("La lunghezza media delle frasi in termini di token del secondo testo é: " + str(len_med_phr_2))

    # comparazione testi in base alla lunghezza media sopracitata
    if len_med_phr_1 > len_med_phr_2:
        print("La lunghezza media delle frasi in termini di token del primo testo é maggiore di quella del secondo testo\n")
    elif len_med_phr_1 < len_med_phr_2:
        print("La lunghezza media delle frasi in termini di token del secondo testo é maggiore di quella del primo testo\n")
    else:
        print("La lunghezza media delle frasi in termini di token é la stessa per entrambi i testi\n")
    
    # calcolo lunghezza media delle parole in termini di caratteri
    len_med_parole_1 = len_med_word(tok_1, nr_tok_1) # calcolo media delle parole in termini di caratteri richiamando funzione
    len_med_parole_2 = len_med_word(tok_2, nr_tok_2)

    print("La lunghezza media delle parole in termini di caratteri del primo testo é: " + str(len_med_parole_1))
    print("La lunghezza media delle parole in termini di caratteri del secondo testo é: " + str(len_med_parole_2))

    # comparazione testi in base alla lunghezza media sopracitata
    if len_med_parole_1 > len_med_parole_2:
        print("La lunghezza media delle parole in termini di caratteri del primo testo é maggiore di quella del secondo testo\n\n")
    elif len_med_parole_1 < len_med_parole_2:
        print("La lunghezza media delle parole in termini di caratteri del secondo testo é maggiore di quella del primo testo\n\n")
    else:
        print("La lunghezza media delle parole in termini di caratteri é la stessa per entrambi i testi\n\n")


    print("3) Grandezza del vocabolario e distribuzione degli hapax all'aumentare del corpus per porzioni incrementali di 1000 token\n")

    # grandezza del vocabolario dell'intero testo
    len_voc_1 = len(set(tok_1))
    len_voc_2 = len(set(tok_2))

    print("La grandezza del vocabolario del primo testo é di " + str(len_voc_1) + " token")
    print("La grandezza del vocabolario del secondo testo é di " + str(len_voc_2) + " token")

    # comparazione testi in base alla grandezza del vocabolario
    if len_voc_1 > len_voc_2:
        print("La grandezza del vocabolario del primo testo é maggiore rispetto a quella del secondo testo\n")
    elif len_voc_1 < len_voc_2:
        print("La grandezza del vocabolario del secondo testo é maggiore rispetto a quella del primo testo\n")
    else:
        print("La grandezza del vocabolario é la stessa per entrambi i testi\n")
    
    # calcolo distribuzione di vocabolario e hapax all'aumentare del corpus per porzioni incrementali di 1000 token
    
    # distribuzione vocabolario e hapax primo testo
    print("Per il primo testo:\n")

    # distribuzione vocabolario
    for type in range(1000, nr_tok_1, 1000): # indicazione intervalli da considerare: range(punto iniziale intervallo, funto finale intervallo, passo di avanzamento)
        distr_voc_1 = distr_voc(tok_1, type) # calcolo distribuzione vocaboli richiamando funzione
        print("La distribuzione delle parole tipo a " + str(type) + " token é di: " + str(distr_voc_1) + " token")
    
    print("\n")

    # distribuzione hapax 
    for hap in range(1000, nr_tok_1, 1000):
        distr_hap_1 = distr_hapax(tok_1, hap) # calcolo distribuzione hapax richiamando funzione
        print("La distribuzione degli hapax a " + str(hap) + " token é di: " + str(distr_hap_1) + " token")


    # distribuzione vocabolario e hapax secondo testo
    print("\nPer il secondo testo:\n")

    # distribuzione vocabolario
    for type in range(1000, nr_tok_2, 1000):
        distr_voc_2 = distr_voc(tok_2, type)
        print("La distribuzione delle parole tipo a " + str(type) + " token é di: " + str(distr_voc_2) + " token")
    
    print("\n")

    # distribuzione hapax 
    for hap in range(1000, nr_tok_2, 1000):
        distr_hap_2 = distr_hapax(tok_2, hap)
        print("La distribuzione degli hapax a " + str(hap) + " token é di: " + str(distr_hap_2) + " token")
    
    print("\n")


    print("4) Rapporto tra sostantivi e verbi\n")

    # annotazione token
    tok_annotati_1 = nltk.pos_tag(tok_1)
    tok_annotati_2 = nltk.pos_tag(tok_2)

    # calcolo rapporto sostantivi-verbi richiamando funzione
    rapporto_sv_1 = rapporto_sv(tok_annotati_1)
    rapporto_sv_2 = rapporto_sv(tok_annotati_2)

    print("Il rapporto tra sostantivi e verbi nel primo testo é: " + str(rapporto_sv_1))
    print("Il rapporto tra sostantivi e verbi nel secondo testo é: " + str(rapporto_sv_2))

    # comparazione testi in base al rapporto sopracitato
    if rapporto_sv_1 > rapporto_sv_2:
        print("Il rapporto tra sostantivi e verbi nel primo testo é maggiore di quello del secondo\n\n")
    elif rapporto_sv_1 < rapporto_sv_2:
        print("Il rapporto tra sostantivi e verbi nel secondo testo é maggiore di quello del primo\n\n")
    else:
        print("Il rapporto tra sostantivi e verbi é lo stesso per entrambi i testi\n\n")
    

    print("5) Le 10 Part of Speech più frequenti\n")

    # calcolo PoS per ogni testo richiamando funzione
    tot_pos_1 = solo_PoS(tok_annotati_1)
    tot_pos_2 = solo_PoS(tok_annotati_2)

    ten_pos_1 = nltk.FreqDist(tot_pos_1).most_common(10) # calcolo distribuzione di frequenza discendente e selezione primi 10 risultati
    ten_pos_2 = nltk.FreqDist(tot_pos_2).most_common(10)
      
    print("Per il primo testo, le 10 PoS più frequenti sono:\n")
    for coppia in ten_pos_1:
        print("PoS " + str(coppia[0]) + " con frequenza " + str(coppia[1]))

    print("\n")

    print("Per il secondo testo, le 10 PoS più frequenti sono:\n")
    for coppia in ten_pos_2:
        print("PoS " + str(coppia[0]) + " con frequenza " + str(coppia[1]))

    print("\n")


    print("6) Estrazione e ordinamento dei 10 bigrammi di PoS con probabilità condizionata e forza associativa massime e relative probabilità\n")

    bigr_pos_1 = bigrams(tot_pos_1) # calcolo bigrammi di PoS
    bigr_pos_2 = bigrams(tot_pos_2)

    ten_bgr_pos_prob_cond_max_1, ten_bgr_pos_lmi_max_1 = ten_bgr_pos_max(tot_pos_1, bigr_pos_1) # calcolo probabilità condizionata e forza associativa massime di dieci bigrammi PoS richiamando funzione
    ten_bgr_pos_prob_cond_max_2, ten_bgr_pos_lmi_max_2 = ten_bgr_pos_max(tot_pos_2, bigr_pos_2) 

    # per il primo testo 
    print("Nel primo testo, i 10 bigrammi di PoS con probabilità condizionata massima sono:\n")
    for coppia in ten_bgr_pos_prob_cond_max_1:
        print("Bigramma PoS " + str(coppia[1]) + ", valore probabilità condizionata: " + str(coppia[0]))

    print("\n")

    print("I 10 bigrammi di PoS con forza associativa massima sono:\n")
    for coppia in ten_bgr_pos_lmi_max_1:
        print("Bigramma PoS " + str(coppia[1]) + ", valore forza associativa: " + str(coppia[0]))
    
    print("\n")

    # per il secondo testo 
    print("Nel secondo testo, i 10 bigrammi di PoS con probabilità condizionata massima sono:\n")
    for coppia in ten_bgr_pos_prob_cond_max_2:
        print("Bigramma PoS " + str(coppia[1]) + ", valore probabilità condizionata: " + str(coppia[0]))

    print("\n")

    print("I 10 bigrammi di PoS con forza associativa massima sono:\n")
    for coppia in ten_bgr_pos_lmi_max_2:
        print("Bigramma PoS " + str(coppia[1]) + ", valore forza associativa: " + str(coppia[0]))
    
# richiamo funzione principale passandole in input i file scelti
main(sys.argv[1], sys.argv[2])

# Una volta nel path della directory del progetto nel terminale, per avviare il programma, digitare " python3 programma1.py testo1.txt testo2.txt "
# Per vidualizzare il risultato in un file di testo che verrà visulizzato nella directory del progetto, aggiungere " > output1.txt "