
def scrape_topic(search_item,output_file_name, tweet_number):
    #  Configure
    c = twint.Config()
    c.Search = search_item
    c.Store_json = True
    c.Output = f"{output_file_name}.json"
    c.Limit = tweet_number
    c.Language='en'
    # Run
    return twint.run.Search(c)