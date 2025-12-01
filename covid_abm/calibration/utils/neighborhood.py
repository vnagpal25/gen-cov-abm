from enum import Enum


class Neighborhood(Enum):
    SUFFOLK = 0
    BRISTOL = 1
    NORFOLK = 2
    PLYMOUTH = 3
    WORCESTER = 4
    ESSEX = 5
    BARNSTABLE = 6
    BERKSHIRE = 7
    HAMPDEN = 8
    HAMPSHIRE = 9

    @property
    def name(self) -> str:
        return {
            Neighborhood.SUFFOLK: "Suffolk County",
            Neighborhood.BRISTOL: "Bristol County",
            Neighborhood.NORFOLK: "Norfolk County",
            Neighborhood.PLYMOUTH: "Plymouth County",
            Neighborhood.WORCESTER: "Worcester County",
            Neighborhood.ESSEX: "Essex County",
            Neighborhood.BARNSTABLE: "Barnstable County",
            Neighborhood.BERKSHIRE: "Berkshire County",
            Neighborhood.HAMPDEN: "Hampden County",
            Neighborhood.HAMPSHIRE: "Hampshire County",
        }[self]

    @property
    def nta_id(self) -> str:
        return {
            Neighborhood.SUFFOLK: "25025",
            Neighborhood.BRISTOL: "25005",
            Neighborhood.NORFOLK: "25021",
            Neighborhood.PLYMOUTH: "25023",
            Neighborhood.WORCESTER: "25027",
            Neighborhood.ESSEX: "25009",
            Neighborhood.BARNSTABLE: "25001",
            Neighborhood.BERKSHIRE: "25003",
            Neighborhood.HAMPDEN: "25013",
            Neighborhood.HAMPSHIRE: "25015",
        }[self]

    @property
    def population(self) -> int:
        return {
            Neighborhood.SUFFOLK: 703621,
            Neighborhood.BRISTOL: 537023,
            Neighborhood.NORFOLK: 674394,
            Neighborhood.PLYMOUTH: 490318,
            Neighborhood.WORCESTER: 773460,
            Neighborhood.ESSEX: 739578,
            Neighborhood.BARNSTABLE: 207666,
            Neighborhood.BERKSHIRE: 116658,
            Neighborhood.HAMPDEN: 432372,
            Neighborhood.HAMPSHIRE: 136121,
        }[self]

    @property
    def text(self) -> str:
        return (
            f"lives in {self.name}, Massachusetts, a county with a population of {self.population}"
        )
