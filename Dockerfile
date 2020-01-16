FROM julia:0.6

WORKDIR /home

COPY test/REQUIRE /home

RUN for dep in $(cat REQUIRE); do julia -e "Pkg.add(\"${dep}\")"; done

COPY . /home

CMD ["bash", "-l"]
