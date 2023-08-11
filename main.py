from prediction import Prediction, PreProcessing
pr = Prediction()
pp = PreProcessing()


if __name__ == '__main__':
   df = pr.readData('tbl_spider_raw')
   df['title'] = df['title'].apply(pp.caseFolding)
   df['title'] = df['title'].apply(pp.tokenizing().tokenize)
   df['title'] = df['title'].apply(pp.stopWord)
   df['title'] = df['title'].apply(pp.stemming)
   
   df['sentiment'] = pr.predictedOwn(df['title'])
   df.to_sql('tbl_spider_raw_result', con=pr.engine(), index=False, if_exists = 'replace')


   print(df.shape)
   print(df.iloc[1])