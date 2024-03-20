#import statements
import pandas as pd
import import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt


''''Percentage increase per subreddit'''
avg_combined_df_pandas = 'read'
#Read avg_combined_per_subreddit csv
fig = px.bar(
    avg_combined_df_pandas,
    x='subreddit',
    y='percentage_increase',
    title='Percentage Increase in Average Posts by Subreddit (Pre-COVID vs During COVID)',
    labels={'percentage_increase': 'Percentage Increase', 'subreddit': 'Subreddit'},
    color='percentage_increase',
    color_continuous_scale=px.colors.sequential.Viridis
)

# Show the figure
fig.show()


'''Sucide indicators word count in suicide watch subreddit'''
word_count_df_pandas = 'read'
#Read suicide_indicator_per_subreddit csv
fig = px.bar(word_count_df_pandas, x='word', y='counts', title='Word Count in Suicidewatch Subreddit Posts')
fig.show()


'''Word cloud on anxiety subreddit'''
from wordcloud import WordCloud
import matplotlib.pyplot as plt

words_list_df = 'read'
#Readfrom anxiety_wordcloud_df.csv
# Join all words into a single string
words_string = ' '.join(words_list_df['Words'])

# Create a WordCloud object
wordcloud = WordCloud(background_color='white').generate(words_string)

# Display the generated word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


'''UserEngagement analysis'''
posts_per_month_pd = 'read'
#read posts_per_month_csv
#viz posts per month
fig = px.bar(posts_per_month_pd, x='month', y='count', title='Post Count per Month in 2019')
fig.show()

posts_by_author_pd ='read'
#read posts_by_author_csv
# Visualize top N authors for simplicity
N = 10
fig = px.bar(posts_by_author_pd.head(N), x='author', y='count', title='Top Authors by Post Count in 2019')
fig.show()


#read posts_per_subreddit.csv
posts_per_month_subreddit_pd ='read'
# Plotly Express visualization
fig = px.line(posts_per_month_subreddit_pd, x='month', y='count', color='subreddit', title='Post Frequency Over Time by Subreddit')
fig.show()


#read distribution_subreddit.csv
posts_by_subreddit_pd = 'read'
fig = px.pie(posts_by_subreddit_pd, names='subreddit', values='count', title='Distribution of Posts by Subreddit')
fig.show()

#read weekly_posts.csv
posts_by_day_subreddit_pd ='read'
fig = px.bar(posts_by_day_subreddit_pd, x='day_of_week', y='count', color='subreddit', title='Post Count by Day of the Week in Each Subreddit')
fig.show()
