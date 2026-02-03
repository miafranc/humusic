# Hungarian popular music lyrics experiments

## Genre categorization with multinomial naive Bayes


Multiclass and one-vs-one classification:
```
python3 tc.py --multiclass
python3 tc.py --ovo
```

## Lyrics complexity

```
python3 complexity.py [--fk|--gzip] [--genres|--playcount]
```
where
- `fk` - Flesch-Kincaid readability
- `gzip` - gzip compression ratio
- `genres` - complexity for genres
- `playcount` - complexity vs playcount plot

## Sentiment analysis

```
 python3 sentiment.py [--idf]
```
- `idf` - use inverse document frequency weighting
