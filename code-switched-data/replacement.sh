find ./split-data/* | parallel --gnu --progress -j 12 python random_replacement_1st.py --file-path \{\} --output-dir panlex-muse-wiki-data-0.5-0.1/cs-data/
