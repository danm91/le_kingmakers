def test_scraper(scrape_str,output_filename,number_tweets):
    #  Configure
    c = twint.Config()
    c.Search = scrape_str
    c.Store_json = True
    c.Output = f"{output_filename}.json"
    c.Limit = number_tweets
    c.Language='en'
    # Run
    return twint.run.Search(c)