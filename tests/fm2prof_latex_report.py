from typing import Dict, List, Optional, Set, Tuple

from ReportGenerator.report_content import ReportContent


class Fm2ProfLatexReport(ReportContent):
    @property
    def project_version(self) -> str:
        return "1.3"

    @property
    def project_name(self) -> str:
        return "FM2Prof"

    @property
    def project_number(self) -> str:
        return "11202219-002"

    @property
    def authors_list(self) -> List[str]:
        return ["Koen Berends", "Asako Fujisaki", "Carles Salvador Soriano Perez"]

    @property
    def case_description_dict(self) -> Dict[str, str]:
        return {
            "Case01rectangleKey": Fm2ProfLatexReport.case_01_rectangle_description(),
            "Case02compoundKey": Fm2ProfLatexReport.case_02_compound_description(),
            "Case03threestageKey": Fm2ProfLatexReport.case_03_three_stage_description(),
            "Case04storageKey": Fm2ProfLatexReport.case_04_storage_description(),
            "Case05dykeKey": Fm2ProfLatexReport.case_05_dyke_description(),
            "Case06plassenKey": Fm2ProfLatexReport.case_06_plassen_description(),
            "Case07triangularKey": Fm2ProfLatexReport.case_07_triangular_description(),
            "Case08waalKey": Fm2ProfLatexReport.case_08_waal_description(),
        }

    @staticmethod
    def case_01_rectangle_description() -> str:
        return r"""
            This case test a basic rectangular profile. 
            It is the simplest rectangular case only with a main channel. 
            The width of the channel is 150m and the longitudinal length is 3000m with a slope of 1:3000.
            We test the following items:
            \begin{enumerate}
            \item Correct generation of the rectangular cross-section
            \item The roughness curve in the main channel
            \item The water volume as function of water level
            \end{enumerate}

            The following output is expected
            \begin{enumerate}
            \item The geometry is expected to be approximatly identical to the true rectangular shape. The following deviations are expected due to the 'slope effect':
                \begin{enumerate}
                    \item The first (most upstream) cross-section is expected to be almost identical
                    \item The last cross-section is expected to have a bias near the bed level, whereby the 1D bed level is higher than the analytical (2D) bed level
                    \item In between, we expect a small error near the bed level, which will result a not perfectly rectangular 1D profile, but exhibiting truncated corners.
                \end{enumerate}
            \item The roughness curve is expected to follow a Manning curve with known n
            \item The volume graph is expected to follow 2D results
            \end{enumerate}
            """

    @staticmethod
    def case_02_compound_description() -> str:
        return r"""
            It has a symmetric 50m wide main channel and 50m floodplains on the both sides of the main channel (total width of 150m). The depth of the main channel is uniformly 2m. The longitudinal length is 3000m with a slope of 1:3000.

            We test the following items:
            \begin{enumerate}
            \item Correct generation of a compound cross-section
            \item The roughness curve in the main channel and floodplain
            \item The water volume as function of water level
            \end{enumerate}

            The following output is expected
            \begin{enumerate}
            \item The geometry is expected to be approximatly identical to the true shape. The following deviations are expected due to the 'slope effect':
                \begin{enumerate}
                    \item The first (most upstream) cross-section is expected to be almost identical
                    \item The last cross-section is expected to have a bias near the bed level, whereby the 1D bed level is higher than the analytical (2D) bed level
                    \item In between, we expect a small error near the bed level, which will result a not perfectly rectangular 1D profile in the lowest stage, but exhibiting truncated corners.
                \end{enumerate}
            \item Two roughness curves that follow Manning curves with known n
            \item The volume graph is expected to follow 2D results
            \end{enumerate}
            """

    @staticmethod
    def case_03_three_stage_description() -> str:
        return r"""
            It has a symmetric 50m wide main channel and two different heights floodplains on the both sides of the main channel. The inner part of the floodplain is 2m from the bottom of the main channel, and the outer floodplain is 0.5m higher than the inner floodplain. Each part of the floodplain has 25m in width (total floodplain width is 100m). The longitudinal length is 3000m with a slope of 1:3000.

            We test the following items:
            \begin{enumerate}
            \item Correct generation of a compound (three-stage) cross-section
            \item The roughness curve in the main channel and floodplain
            \item The water volume as function of water level
            \end{enumerate}

            The following output is expected
            \begin{enumerate}
            \item The geometry is expected to be approximatly identical to the true shape. The following deviations are expected due to the 'slope effect':
                \begin{enumerate}
                    \item The first (most upstream) cross-section is expected to be almost identical
                    \item The last cross-section is expected to have a bias near the bed level, whereby the 1D bed level is higher than the analytical (2D) bed level
                    \item In between, we expect a small error near the bed level, which will result a not perfectly rectangular 1D profile in the lowest stage, but exhibiting truncated corners.
                \end{enumerate}
            \item The roughness curve for the main channel is expected to follow a Manning curve with known n. 
            The second (floodplain) curve is expected to be a compound curve, build up from known curves from the two floodplain stages. 
            \item The volume graph is expected to follow 2D results
            \end{enumerate}
            """

    @staticmethod
    def case_04_storage_description() -> str:
        return r"""
            The basic geometry is the same as Case 02. 
            In addition, infinitely high walls (a.k.a. thin dams in FM) are placed at 1250 and 1750m from the inlet perpendicular to the flow direction on one side of the floodplain. 
            The area between two walls is considered a storage area because the flow is significantly slower than others.

            We test the following items:
            \begin{enumerate}
            \item Correct generation of a compound (two-stage) cross-section
            \item The roughness curve in the main channel and floodplain
            \item The water volume as function of water level
            \item At cross-section 1500, correct estimation of the width of the storage
            \end{enumerate}

            The following output is expected
            \begin{enumerate}
            \item For volume, rougness and geometry, this case follows case 2, with the exception of:
            \item At cross-section 1500, the flow width should be smaller than the total width. 
            \end{enumerate}
            """

    @staticmethod
    def case_05_dyke_description() -> str:
        return r"""
            The basic geometry is the same as Case 02, but there is compartimentalistation due the embankments ('summer dikes'). 
            At the boundary of the main channel and the floodplain, there are 1m high embankments on the both side of the main channel.  
            Therefore, the water flows into the floodplain after the water depth reaches 3m instead of 2m in Case 02.

            Expected outcome:
            We test the following items:
            \begin{enumerate}
            \item Correct generation of a compound (two-stage) compartimentalised cross-section.
            \item The water volume as function of water level
            \end{enumerate}

            The following output is expected
            \begin{enumerate}
            \item The generated cross-section should be comparable to case 2, but with the floodplain level at the higher of the embankment. 
            \item The volume graphs should show that the applied volume correction is able to correctly follow 2D results. 
            \end{enumerate}

            \Note{The roughness values are shown for this case, but it is not (yet) known which values should be expected in this case.}

            """

    @staticmethod
    def case_06_plassen_description() -> str:
        return r"""
            The basic geometry is the same as Case 02. 
            A deep lake is located on the floodplain between 1250m and 1750m of the domain. 
            The width of the lake is 25m, and it takes up the outer half of one side of the floodplain. 
            Although the depth of the lake is approximately 10m from the floodplain, it does not influence the cross-section but roughness.

            Expected outcome:
            We test the following items:
            \begin{enumerate}
            \item Correct generation of a compound (two-stage) compartimentalised cross-section.
            \item The roughness curve in the main channel and floodplain
            \end{enumerate}

            The following output is expected
            \begin{enumerate}
            \item The generated cross-section should be comparable to case 2. The lake should not affect geometry.
            \item At cross-section 1500, the roughness should be high at water level under the floodplain level, then sharply decrease as the water level exceeds the floodplain level. 
            \end{enumerate}


            """

    @staticmethod
    def case_07_triangular_description() -> str:
        return r"""
            It has a triangular 2D grid in FM instead of rectangular 2D grid like the rest of the cases, and the overall geometry is similar to Case 02. 
            The total width of the domain is 500m and the length is 10000m. 
            The most part of the main channel has the width of 200m, and the floodplain has 150m at each side; however, the main channel width is increased to 250m near the inlet and outlet of the domain due to the geometrical limitation to represent the rectangular domain with “almost” regular triangles. 
            It has a slope of 1:5000.

            We test the following items:
            \begin{enumerate}
            \item Correct generation of the rectangular cross-section
            \item The roughness curve in the main channel
            \item The water volume as function of water level
            \end{enumerate}

            The following output is expected
            \begin{enumerate}
            \item The same expectation as case 2. However, inaccuracies are expected due to the inefficient triangular grid given rectangular geometry. 
            \end{enumerate}
            """

    @staticmethod
    def case_08_waal_description() -> str:
        return r"""
            The Waal case is a 'real world' study case of the River Waal. 
            In this case, the 'true' geometry and roughness are not known. 
            In this test, we directly compare 1D model results with 2D model results. 

            We test the following items:
            \begin{enumerate}
            \item Whether output generated by FM2PROF can be succesfully used as input for the 1D model
            \item The results of the 1D model compared to results of the 2D model at selected locations. 
            \end{enumerate}

            The following output is expected
            \begin{enumerate}
            \item A graph showing the overall statistics of deviations between 1D and 2D model results
            \item Graphs of water level over time, showing 1D and 2D model results, as well as the deviation between them. 
            \end{enumerate}

            """
