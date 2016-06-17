import math
from build_model import readResponseData, readData, readTimeDf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("darkgrid")
sns.set_context("poster", font_scale=1.3)

starting_date = "2016-02-01"


content_df = readTimeDf(starting_date=starting_date)

content_df = content_df.drop_duplicates('id')

content_df = content_df[content_df['ElapsedTime'] < 24 * 60]
content_df = content_df[content_df['ElapsedTime'] > 0]


plt.figure()
content_df.ElapsedTime.hist(bins=50)
plt.title("Distribution of the elapsed time for accepted answers")
plt.xlabel('Time [Min]')
plt.savefig("good_answer_time.png")

plt.figure()
content_df.ElapsedTime.apply(math.log).hist(bins=50)
plt.title("Distribution of the elapsed time for accepted answers")
plt.xlabel('Logarithm of Time [Min]')
plt.savefig("good_answer_log_time.png")

# q_a, questions, questions_unanswered_ids = readResponseData(
#     starting_date=starting_date)


# shortest_elapsed_time = q_a.groupby('id_x').apply(
#     lambda g: g['ElapsedTime'].min())

# plt.figure()
# shortest_elapsed_time.hist(bins=50)
# plt.xlabel('')
# plt.savefig("first_answer_hist.png")

# print("One day = {} min".format(24 * 60))

# questions_answered_late_ids = shortest_elapsed_time[shortest_elapsed_time >
#                                                     24 * 60].index

# failed_questions_ids = np.concatenate((questions_unanswered_ids.values,
#                                         questions_answered_late_ids))

# questions['success'] = questions['id'].isin(failed_questions_ids).apply(
#     lambda b: 0 if b else 1)

# cls_df = questions[feature_cols + ['id', 'success']]
