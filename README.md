# Lab Samples Logistics

## Instruções para reprodução da síntese de controlador para missão

Em sistemas Linux:

### Passo 1: Instalação da ferramenta _slugs_

Seguir instruções do repositório [_slugs_](https://github.com/VerifiableRobotics/slugs):

```bash
$ git clone https://github.com/VerifiableRobotics/slugs
$ cd src
$ make
```
### Passo 2: Síntese do controlador

Para compilar os requisitos LTL para um formato compatível com a ferramenta, usar o parser disponível no repositório:

```bash
$ python3 slugs/tools/StructuredSlugsParser/compiler.py LabSamples.structuredslugs > LabSamples.slugsin
```

Usar a ferramenta:

```bash
$ ./slugs/src/slugs --explicitStrategy --jsonOutput LabSamples.slugsin > controller.json
```

### Passo 3: Análise do Controlador

```bash
$ python3 controller_test.py
```
Selecionar modo de simulação seguindo as instruções do programa.

