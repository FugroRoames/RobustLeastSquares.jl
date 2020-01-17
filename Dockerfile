FROM julia:1.0

WORKDIR /home

COPY test/REQUIRE /home

RUN for dep in $(cat REQUIRE); do julia -e "using Pkg; Pkg.add(\"${dep}\")"; done

COPY . /home

CMD ["bash", "-l"]
