""" Database models for microstructure dataset """

import os
from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

Base = declarative_base()

dbpath = 'sqlite:///data/microstructures.sqlite'

class User(Base):
    __tablename__ = 'user'
    user_id = Column(Integer, primary_key=True)
    username =  Column(String(250))
    givenname = Column(String(250))
    familyname =  Column(String(250))
    email =     Column(String(250))
    orcid = Column(String(250))
    micrographs = relationship('Micrograph')
        
class Collection(Base):
    __tablename__ = 'collection'
    collection_id =   Column(Integer, primary_key=True)
    name = Column(String(250))
    doi = Column(String(250))
    
class Sample(Base):
    __tablename__ = 'sample'
    sample_id = Column(Integer, primary_key=True)
    label = Column(String(250))
    anneal_time = Column(Float)
    anneal_time_unit = Column(String(16))
    anneal_temperature = Column(Float)
    anneal_temp_unit = Column(String(16))
    cool_method = Column(String(16))
    micrographs = relationship('Micrograph')
    
class Micrograph(Base):
    __tablename__ = 'micrograph'
    micrograph_id = Column(Integer, primary_key=True)
    path =             Column(String())
    micron_bar =       Column(Float)
    micron_bar_units = Column(String(64))
    micron_bar_px =    Column(Integer)
    magnification =    Column(Integer)
    detector =         Column(String(16))
    sample_key =        Column(Integer, ForeignKey('sample.sample_id'))
    sample =           relationship('Sample', back_populates='micrographs')
    contributor_key =   Column(Integer, ForeignKey('user.user_id'))
    contributor =      relationship('User', back_populates='micrographs')
    primary_microconstituent = Column(String(250))


if __name__ == '__main__':
    engine = create_engine(dbpath)

    Base.metadata.create_all(engine)
