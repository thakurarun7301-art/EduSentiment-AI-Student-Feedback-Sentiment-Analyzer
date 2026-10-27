import pandas as pd
import random

positive = [
    "The teacher explains concepts very clearly.",
    "The teacher explains every topic very well.",
    "The lectures are interesting and useful.",
    "The practical classes are very helpful.",
    "I enjoyed the practical sessions.",
    "The teacher gives useful examples.",
    "The course material is easy to understand.",
    "The teacher answers questions patiently.",
    "The classes are interactive.",
    "I am satisfied with the teaching.",
    "The course improved my knowledge.",
    "The teacher is very supportive.",
    "The assignments are useful for learning.",
    "The lectures are informative.",
    "The practical work is interesting.",
    "The teacher motivates students.",
    "The course is well organized.",
    "The explanations are simple.",
    "I enjoyed attending the classes.",
    "The teaching quality is excellent.",
    "The teacher provides helpful feedback.",
    "The concepts became easy to understand.",
    "The classroom environment is good.",
    "The study material is useful.",
    "I learned many new things.",
    "The instructor explains difficult topics well.",
    "The course is engaging.",
    "The teacher teaches with good examples.",
    "The practical work improved my skills.",
    "The lectures are easy to follow.",
    "The teacher clears doubts properly.",
    "The course content is interesting.",
    "The teaching method is effective.",
    "The classes are enjoyable.",
    "The teacher gives clear instructions.",
    "The assignments help me practice.",
    "The subject is explained properly.",
    "The learning experience is good.",
    "The teacher is approachable.",
    "The course is helpful."
]

neutral = [
    "The classes are regular.",
    "The course content is average.",
    "The teaching is okay.",
    "The lectures are normal.",
    "The course is acceptable.",
    "The practical classes are average.",
    "The teacher explains the main topics.",
    "The study material is acceptable.",
    "The classes are manageable.",
    "The teaching quality is average.",
    "The instructor is okay.",
    "The assignments are normal.",
    "The classroom experience is average.",
    "The lectures are sometimes interesting.",
    "The course is neither easy nor difficult.",
    "The teacher provides basic explanations.",
    "The classes are conducted regularly.",
    "The overall experience is satisfactory.",
    "The teaching method is standard.",
    "The lectures are regular.",
    "The course meets basic expectations.",
    "The teacher covers the required topics.",
    "The practical work is normal.",
    "The assignments are manageable.",
    "The class experience is ordinary.",
    "The teacher gives basic feedback.",
    "The course has easy and difficult topics.",
    "The lectures are conducted as scheduled.",
    "The study material is sufficient.",
    "The teaching is neither good nor bad.",
    "The class was average.",
    "The course is fairly normal.",
    "The instructor teaches the basic concepts.",
    "The practical sessions are regular.",
    "The course structure is standard.",
    "The lectures are okay.",
    "The teacher explains some topics well.",
    "The learning experience is average.",
    "The course is reasonably organized.",
    "The class environment is normal."
]

negative = [
    "The teacher explains concepts poorly.",
    "The teaching is difficult to understand.",
    "The lectures are boring.",
    "The practical classes are not useful.",
    "The teacher does not explain properly.",
    "I am not satisfied with the teaching.",
    "The study material is confusing.",
    "The classes are difficult to follow.",
    "I did not enjoy the lectures.",
    "The course content is poorly explained.",
    "The teacher does not answer questions clearly.",
    "The assignments are confusing.",
    "The classroom experience is disappointing.",
    "The lectures are not interesting.",
    "I have difficulty understanding the concepts.",
    "The teaching method is ineffective.",
    "The practical sessions are boring.",
    "The teacher needs to explain topics better.",
    "The course is poorly organized.",
    "I am unhappy with the teaching quality.",
    "The explanations are unclear.",
    "The course material is difficult to understand.",
    "The classes are not engaging.",
    "I did not find the course helpful.",
    "The teacher explains topics too quickly.",
    "The lectures are hard to understand.",
    "The practical work is disappointing.",
    "The teacher does not clear doubts.",
    "The course is difficult to follow.",
    "The teaching quality is poor.",
    "The classes are boring.",
    "The instructor is not helpful.",
    "The examples are unclear.",
    "The course material is confusing.",
    "The teacher gives insufficient explanations.",
    "The assignments are difficult to understand.",
    "The lectures lack useful examples.",
    "The practical classes are ineffective.",
    "The learning experience is frustrating.",
    "The course does not help me understand concepts."
]


def expand_list(data, label, target=100):
    result = []

    for text in data:
        result.append((text, label))

    while len(result) < target:
        text = random.choice(data)
        result.append((text, label))

    return result[:target]


positive_data = expand_list(positive, "positive")
neutral_data = expand_list(neutral, "neutral")
negative_data = expand_list(negative, "negative")

all_data = positive_data + neutral_data + negative_data

random.shuffle(all_data)

df = pd.DataFrame(
    all_data,
    columns=["text", "label"]
)

df.to_csv(
    "data/feedback.csv",
    index=False
)

print("Dataset created successfully!")
print()
print("Total rows:", len(df))
print()
print("Dataset distribution:")
print(df["label"].value_counts())
print()
print("Saved to: data/feedback.csv")