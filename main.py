from GoogleNews import GoogleNews
import pandas as pd
import newspaper

googlenews = GoogleNews(start='01/01/2014', end='11/22/2021')
googlenews.setlang('pt')
googlenews.search('Nintendo')

for i in range(2, 5):
    googlenews.getpage(i)
    result = googlenews.result()
    df = pd.DataFrame(result)

print(df)

for ind in df.index:
    dict={}
    article = newspaper.Article(df['link'][ind])
    article.download()
    article.parse()
    article.nlp()
    dict['Date'] = df['date'][ind]
    dict['Media'] = df['media'][ind]
    dict['Title'] = article.title
    dict['Article'] = article.text
    dict['Summary'] = article.summary
    list.append(dict)
news_df = pd.DataFrame(list)
# news_df.to_excel("articles.xlsx")

print(news_df[1]['Article'])
