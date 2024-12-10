import json
from dataclasses import dataclass, field, asdict, astuple
from typing import List, Union, Any, Type, TypeVar

T = TypeVar('T', bound='YoloBase')


@dataclass
class YoloBase:
    """基类，提供通用的序列化和反序列化方法。"""
    
    def to_list(self) -> List[Any]:
        return astuple(self)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_list(cls: Type[T], data: List[Any]) -> T:
        return cls(*data)
    
    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        return cls(**data)
    
    @classmethod
    def from_json(cls: Type[T], data: str) -> Union[T, List[T]]:
        try:
            data_parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON data") from e
        
        if isinstance(data_parsed, list):
            return [cls.from_dict(item) for item in data_parsed]
        elif isinstance(data_parsed, dict):
            return cls.from_dict(data_parsed)
        else:
            raise TypeError("JSON must represent a dict or a list of dicts")


@dataclass
class Yolo(YoloBase):
    lx: int = 0
    ly: int = 0
    rx: int = 0
    ry: int = 0
    cls: int = 0
    conf: float = 0.0


@dataclass
class YoloPoint(YoloBase):
    x: int = 0
    y: int = 0
    conf: float = 0.0


@dataclass
class YoloPose(Yolo):
    pts: List[YoloPoint] = field(default_factory=lambda: [YoloPoint() for _ in range(17)])
    
    def to_list(self) -> List[Any]:
        base_list = super().to_list()
        pts_list = [pt.to_list() for pt in self.pts]
        return base_list + [pts_list]
    
    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict['pts'] = [pt.to_dict() for pt in self.pts]
        return base_dict
    
    @classmethod
    def from_list(cls, data: List[Any]) -> 'YoloPose':
        if len(data) != 7:
            raise ValueError("List must contain exactly 7 elements for YoloPose")
        *base_data, pts_data = data
        pts = [YoloPoint.from_list(pt) for pt in pts_data]
        return cls(*base_data, pts=pts)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'YoloPose':
        pts_data = data.pop('pts', [])
        pts = [YoloPoint.from_dict(pt) for pt in pts_data]
        return cls(**data, pts=pts)
