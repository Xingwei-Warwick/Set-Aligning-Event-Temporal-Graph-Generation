"""
Source: https://github.com/ConstantineLignos/nyt-corpus-reader
Parse New York Times Annotated Corpus files.
"""

import datetime
# noinspection PyPep8Naming
import xml.etree.ElementTree as ET
from typing import Sequence, Iterable, Optional, Dict, Any, Union, TextIO

from attr import attrs, attrib, asdict

NO_INDEX_TERMS = 'NO INDEX TERMS FROM NYTIMES'


@attrs
class NYTArticle:
    """
    Parse and store the fields of an NYT Annotated Corpus article.

    Note that due to issues with the original data, descriptors,
    general descriptors, and types of material are lowercased. As
    some types of material mistakenly contain article text, long
    entries or entries containing tags in that field are removed.
    """

    docid: str = attrib()
    title: Optional[str] = attrib()
    date: datetime.datetime = attrib()
    descriptors: Sequence[str] = attrib()
    general_descriptors: Sequence[str] = attrib()
    types_of_material: Sequence[str] = attrib()
    paragraphs: Sequence[str] = attrib()

    @classmethod
    def from_element_tree(cls, root: Union[ET.Element, ET.ElementTree]) -> 'NYTArticle':
        head = root.find("./head")
        title_element = head.find("./title")
        title = title_element.text if title_element is not None else None
        # pubdata appears to always be there, but publication_{year,month,day_of_month} are missing
        # in some articles
        pubdata = head.find("./pubdata")
        date = datetime.datetime.strptime(pubdata.get('date.publication'), '%Y%m%dT%H%M%S')

        docdata = head.find("./docdata")
        docid = docdata.find("./doc-id").get('id-string')
        assert docid is not None
        descriptors = _clean_descriptors(d.text for d in docdata.findall(
            "./identified-content/*[@type='descriptor']"))
        general_descriptors = _clean_descriptors(d.text for d in docdata.findall(
            "./identified-content/*[@type='general_descriptor']"))
        types_of_material = _clean_types_of_material(d.text for d in docdata.findall(
            "./identified-content/classifier[@type='types_of_material']"))

        paragraphs = [p.text for p in root.findall(
            "./body/body.content/*[@class='full_text']/p")]

        # Mypy and Pycharm don't understand the attrs __init__ arguments
        # noinspection PyArgumentList
        return cls(docid, title, date, descriptors, general_descriptors,  # type: ignore
                   types_of_material, paragraphs)

    @classmethod
    def from_file(cls, input_file: TextIO) -> 'NYTArticle':
        return cls.from_element_tree(ET.parse(input_file))

    @classmethod
    def from_str(cls, contents: str) -> 'NYTArticle':
        return cls.from_element_tree(ET.fromstring(contents))

    def as_dict(self) -> Dict[Any, Any]:
        return asdict(self)


def _clean_descriptors(descriptors: Iterable[str]) -> Sequence[str]:
    """Deduplicate and clean descriptors

    The returned sequence is sorted to ensure the ordering is deterministic.
    """
    return sorted(set(descriptor.lower() for descriptor in descriptors
                      if descriptor is not None and descriptor != NO_INDEX_TERMS))


def _clean_types_of_material(types_of_material: Iterable[str]) -> Sequence[str]:
    """Remove any bad data in types_of_material"""
    # We use the presence of newline or <> (for tags) as evidence that something is wrong
    return [item.lower() for item in types_of_material
            if item is not None and len(item) < 50 and '\n' not in item
            and '<' not in item and '>' not in item]