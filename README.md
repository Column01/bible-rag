# Bible RAG

Simple CLI tool for retreiving scripture via text embedings (future plan to add LLM agents in the mix)

## Installation and usage

1. Install Python 3.11+
2. `pip install .` in the root of the repository
3. `bible-rag --help`

You must run `bible-rag --setup` at least once to setup the project for searching. 

You can limit it to indexing your preferred translation by using the `--translation` flag with any translation code listed in `bible-rag --list-translations` (make sure you have terminal that can scroll)

## Example

```bash
bible-rag -t ESV -s "Passages relating to the Son of Man in Daniel and in the Gospels" -n 10   
{'book': 'Daniel', 'chapter': '7', 'verse': '13', 'text': '"I saw in the night visions, and behold, with the clouds of heaven there came one like a son of man, and he came to the Ancient of Days and was presented before 
him.', 'distance': 0.22948765754699707}
{'book': 'Matthew', 'chapter': '16', 'verse': '13', 'text': 'Now when Jesus came into the district of Caesarea Philippi, he asked his disciples, "Who do people say that the Son of Man is?"', 'distance': 0.2528941035270691}
{'book': 'Luke', 'chapter': '9', 'verse': '44', 'text': '"Let these words sink into your ears: The Son of Man is about to be delivered into the hands of men."', 'distance': 0.25377464294433594}
{'book': 'Luke', 'chapter': '17', 'verse': '26', 'text': 'Just as it was in the days of Noah, so will it be in the days of the Son of Man.', 'distance': 0.2545093894004822}
{'book': 'Matthew', 'chapter': '17', 'verse': '22', 'text': 'As they were gathering in Galilee, Jesus said to them, "The Son of Man is about to be delivered into the hands of men,', 'distance': 0.25719624757766724}      
{'book': 'Luke', 'chapter': '6', 'verse': '5', 'text': 'And he said to them, "The Son of Man is lord of the Sabbath."', 'distance': 0.25822919607162476}
{'book': 'Ezekiel', 'chapter': '27', 'verse': '2', 'text': '"Now you, son of man, raise a lamentation over Tyre,', 'distance': 0.25890302658081055}
{'book': 'Ezekiel', 'chapter': '28', 'verse': '21', 'text': '"Son of man, set your face toward Sidon, and prophesy against her', 'distance': 0.26247817277908325}
{'book': 'Daniel', 'chapter': '10', 'verse': '16', 'text': 'And behold, one in the likeness of the children of man touched my lips. Then I opened my mouth and spoke. I said to him who stood before me, "O my lord, by reason of the vision pains have come upon me, and I retain no strength.', 'distance': 0.26260238885879517}
{'book': 'Ezekiel', 'chapter': '16', 'verse': '2', 'text': '"Son of man, make known to Jerusalem her abominations,', 'distance': 0.26514875888824463}
```
