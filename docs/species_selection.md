# Species and region

**Chosen: *Vespa velutina* (Asian hornet, GBIF backbone key 1311477) in Western Europe.**

The candidate-evaluation step from the original brief was skipped at the user's request: the user
said to use either *Aedes albopictus* in Europe or *Vespa velutina* in Western Europe. To pick
between the two, we ran one GBIF occurrence-search facet query per species (Europe,
`hasCoordinate=true`, `hasGeospatialIssue=false`) on 2026-10-24:

| | *Vespa velutina* | *Aedes albopictus* |
|---|---|---|
| Georeferenced European records | 464,439 | 78,832 |
| Years with >= 500 records | 2006-2026 (every year) | 2015-2026 (most years) |
| Main countries | BE, NL, FR, CH, DE, ES, PT | AT, ES, IT, FR, HU |

*V. velutina* has about 6x more records. It also has a well-documented single point of
introduction (south-west France, 2004) and a front that has kept moving since then into Spain,
Portugal, Belgium, the Netherlands, Germany, Switzerland and the Channel Islands/UK. Both fit
the brief's requirement of at least ~5,000 records over at least 8 years.

**Region** (`configs/default.yaml`): PT, ES, AD, FR, MC, BE, NL, LU, DE, CH, LI, GB, IE, JE, GG
and IM, clipped to the bbox [-11.0, 35.5, 15.5, 61.0], which leaves out Macaronesia.
Italy and Austria are excluded to keep the region compact; Italy has only a few hundred records
(Liguria). Both can be added in the config.
