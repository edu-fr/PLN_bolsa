import sys
from GoogleNews import GoogleNews
import pandas as pd
from newspaper import Config
import yfinance as yf
from newspaper import Article
from os.path import exists
import nltk
import re

#  Primeiro criamos uma base de treino usando noticias aleatorias -> Como avaliar elas como boas ou ruins? Usamos
#  a classificação pronta do SentiLex-lem-PT02.txt
# A partir disso temos uma lista de noticias classificadas como boas ou ruins.


def main():

    # Arguments:
    args = sys.argv[1:]
    update_google = args[0]  # True or False
    initial_date = args[1]  # mm/dd/yyyy
    final_date = args[2]  # mm/dd/yyyy
    pages = int(args[3])  # int (ex: 5)
    period = args[4]  # str (ex: '5d')
    first_time = args[5]  # True or False

    # Quando for rodar pela primeira vez
    if first_time == "True":
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('stopwords')

    if update_google == "True":
        update_google = True
    else:
        update_google = False

    if update_google:
        print("Atualizando arquivos por opção")
        updateData(update_google, starting_date=initial_date, ending_date=final_date, pages=pages)
    elif not exists('export_dataframe.csv'):
        print("Dataframe final não encontrado. Atualizando arquivos")
        updateData(update_google, starting_date=initial_date, ending_date=final_date, pages=pages)

    print("Importando do csv")
    final_df = pd.read_csv('export_dataframe.csv', '&')

    final_df.drop('Unnamed: 0', axis='columns', inplace=True)
    # print(final_df)

    #print(final_df['Article'][0])

    #finance_data = yf.download(tickers='NTDOY', period=period, interval='1d')
    #print(finance_data)

    # TODO: Dar nota para os artigos; Cruzar a data dos artigos com o valor da ação daqueles dias;
    preprocess(final_df)

    database_sentiment = set_database_sentiment()

    x = 0
    for i in final_df['Article']:
        final_df['Score'][x] = evaluate_sentiment(i, database_sentiment)
        x = x + 1

    contador = 0
    for j in final_df['Score']:
        print(str(j) + "    Noticia: " + final_df['Title'][contador])
        contador = contador + 1

def updateData(update_google, starting_date, ending_date, pages):
    print("Atualizando dados!")
    if update_google:
        print("Atualizando google por escolha")
        updateGoogle(starting_date, ending_date, pages)
    elif not exists('google_news_df.csv'):
        print("Arquivo com noticias do google nao existe, criando novo arquivo")
        updateGoogle(starting_date, ending_date, pages)

    google_news_dataframe = pd.read_csv(filepath_or_buffer="google_news_df.csv",
                                        sep='&')  # Lê o arquivo com as notícias do google
    google_news_dataframe.drop('Unnamed: 0', axis='columns', inplace=True)

    dicts = []

    # Configure settings to download articles
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) ' \
                 'Chrome/91.0.4472.124  Safari/537.36 '
    config = Config()
    config.browser_user_agent = user_agent

    print("Iniciando download das noticias completas: ")
    for ind in google_news_dataframe.index:
        print("Noticia " + str(ind))
        dict = {}
        article = Article(google_news_dataframe['link'][ind])
        article.download()
        article.parse()
        article.nlp()
        dict['Date'] = google_news_dataframe['date'][ind]
        dict['Media'] = google_news_dataframe['media'][ind]
        dict['Title'] = article.title
        dict['Article'] = article.text
        dict['Summary'] = article.summary
        dict['Score'] = 0
        dicts.append(dict)
    news_df = pd.DataFrame(dicts)
    news_df.to_csv(path_or_buf='export_dataframe.csv', sep='&')  # Exporta para um arquivo com as noticias completas


def updateGoogle(starting_date, ending_date, pages):
    google_news = GoogleNews(start=starting_date, end=ending_date)
    google_news.setlang('pt')
    google_news.search('Nintendo')
    print("Recuperando notícias atraves da API do Google News...")
    for i in range(2, pages):
        google_news.getpage(i)
        result = google_news.result()
        df = pd.DataFrame(result)
        print("Noticias da pagina " + str(i) + " extraidas.")

    pd.set_option("display.max_rows", None, "display.max_columns", None)  # Altera a visualização do dataframe
    df.to_csv(path_or_buf='google_news_df.csv', sep='&')  # Cria um arquivo com as noticias do google


def preprocess(final_df):

    language = 'portuguese'
    final_df['Article'] = final_df.apply(lambda row: nltk.word_tokenize(str(row['Article'])), axis=1)
    stopwords = nltk.corpus.stopwords.words(language)
    stopwords = list(set(stopwords))
    final_df['Article'] = final_df.apply(lambda row: normalize(row['Article'], stopwords), axis=1)

    # print(final_df['Article'][0])


def normalize(words, stopwords):
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words, stopwords)

    return ' '.join(words)


def remove_stopwords(words, stopwords):
    """Remover as Stopwords das palavras tokenizadas"""
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words


def to_lowercase(words):
    """converter todos os caracteres para lowercase"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """remover pontuação"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)

    return new_words


def set_database_sentiment():

    sentilexpt = open('SentiLex-lem-PT02.txt')

    dic_palavra_polaridade = {}
    for i in sentilexpt.readlines():
        pos_ponto = i.find('.')
        palavra = (i[:pos_ponto])
        pol_pos = i.find('POL')
        polaridade = (i[pol_pos + 7:pol_pos + 9]).replace(';', '')
        dic_palavra_polaridade[palavra] = polaridade

    return dic_palavra_polaridade


def evaluate_sentiment(text, database_sentiment):
    score = 0
    text = str(text)
    num_palavras = 0
    num_iterado = 0

    for word in text.split():
        num_palavras = num_palavras + 1
        if word in database_sentiment:
            score = score + int(database_sentiment[word])
            num_iterado = num_iterado + 1

    #print( "Numero de palavras: " + str(num_palavras) + "  Numero iterado: " + str(num_iterado))
    return score

main()
