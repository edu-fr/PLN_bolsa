import sys
from GoogleNews import GoogleNews
import pandas as pd
from newspaper import Config
import yfinance as yf
from newspaper import Article
from os.path import exists

def main():
    # Arguments:
    args = sys.argv[1:]
    update_google = args[0]     # True or False
    initial_date = args[1]      # mm/dd/yyyy
    final_date = args[2]        # mm/dd/yyyy
    pages = int(args[3])        # int (ex: 5)
    period = args[4]            # str (ex: '5d')

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

    print(final_df['Article'][0])

    finance_data = yf.download(tickers='NTDOY', period=period, interval='1d')
    print(finance_data)

    # TODO: Dar nota para os artigos; Cruzar a data dos artigos com o valor da ação daqueles dias;


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


main()
