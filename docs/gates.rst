Quantum gates
=======================

This is a list of gates that qujax supports natively. You can also define custom operations by directly passing an array or function instead of a string, as documented in :doc:`statetensor/get_params_to_statetensor_func` and :doc:`densitytensor/get_params_to_densitytensor_func`.

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Name(s)
     - String
     - Matrix representation
   * - Pauli X gate

       NOT gate

       Bit flip gate
     - :python:`"X"`
     - .. math:: X = NOT = \begin{bmatrix}0 & 1\\ 1 & 0 \end{bmatrix}
   * - Pauli Y gate
     - :python:`"Y"`
     - .. math:: Y = \begin{bmatrix}0 & -i\\ i & 0 \end{bmatrix}        
   * - Pauli Z gate

       Phase flip gate
     - :python:`"Z"`
     - .. math:: Z = \begin{bmatrix}1 & 0\\ 0 & -1 \end{bmatrix}     
   * - Hadamard gate
     - :python:`"H"`
     - .. math:: H = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\ 1 & -1 \end{bmatrix}
   * - S gate

       P (phase) gate
     - :python:`"S"`
     - .. math:: S = P = \sqrt{Z} = \begin{bmatrix}1 & 0\\ 0 & i \end{bmatrix}
   * - Conjugated S gate
     - :python:`"Sdg"`
     - .. math:: S^\dagger = \begin{bmatrix}1 & 0\\ 0 & -i \end{bmatrix}
   * - T gate
     - :python:`"T"`
     - .. math:: T = \sqrt[4]{Z} = \begin{bmatrix}1 & 0\\ 0 & \exp(\frac{\pi i}{4}) \end{bmatrix}
   * - Conjugated T gate
     - :python:`"Tdg"`
     - .. math:: T^\dagger = \begin{bmatrix}1 & 0\\ 0 & -\exp(\frac{\pi i}{4}) \end{bmatrix}
   * - V gate
     - :python:`"V"`
     - .. math:: V = \sqrt{X} = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & -i\\ -i & 1 \end{bmatrix}
   * - Conjugated V gate
     - :python:`"Vdg"`
     - .. math:: V^\dagger = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & i\\ i & 1 \end{bmatrix}
   * - SX gate
     - :python:`"SX"`
     - .. math:: SX = \sqrt{X} = \frac{1}{2}\begin{bmatrix}1 + i & 1 - i\\ 1 - i & 1 + i \end{bmatrix}
   * - Conjugated SX gate
     - :python:`"SXdg"`
     - .. math:: SX^\dagger = \frac{1}{2}\begin{bmatrix}1 - i & 1 + i\\ 1 + i & 1 - i \end{bmatrix}
   * - CX (Controlled X) gate

       CNOT gate
     - :python:`"CX"`
     - .. math:: CX = CNOT = \begin{bmatrix}I & 0\\ 0 & X \end{bmatrix} = \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0  \end{bmatrix}
   * - CY (Controlled Y) gate
     - :python:`"CY"`
     - .. math:: CY = \begin{bmatrix}I & 0\\ 0 & Y \end{bmatrix} = \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & -i \\ 0 & 0 & i & 0  \end{bmatrix}
   * - Controlled Z gate
     - :python:`"CZ"`
     - .. math:: CZ = \begin{bmatrix}I & 0\\ 0 & Z \end{bmatrix} = \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1  \end{bmatrix}
   * - Controlled Hadamard gate
     - :python:`"CH"`
     - .. math:: CH = \begin{bmatrix}I & 0\\ 0 & H \end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 1 & -1  \end{bmatrix}
   * - Controlled V gate
     - :python:`"CV"`
     - .. math:: CV = \begin{bmatrix}I & 0\\ 0 & V \end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & -i \\ 0 & 0 & -i & 1  \end{bmatrix}
   * - Conjugated controlled V gate
     - :python:`"CVdg"`
     - .. math:: CVdg = \begin{bmatrix}I & 0\\ 0 & V^\dagger \end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & i \\ 0 & 0 & i & 1  \end{bmatrix}
   * - Controlled SX gate
     - :python:`"CSX"`
     - .. math:: CSX = \begin{bmatrix}I & 0\\ 0 & SX \end{bmatrix} = \frac{1}{2}\begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1+i & 1-i \\ 0 & 0 & 1-i & 1+i  \end{bmatrix}
   * - Conjugated controlled SX gate
     - :python:`"CSXdg"`
     - .. math:: CSX^\dagger = \begin{bmatrix}I & 0\\ 0 & SX^\dagger \end{bmatrix} = \frac{1}{2}\begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1-i & 1+i \\ 0 & 0 & 1+i & 1-i  \end{bmatrix}
   * - Toffoli gate

       CCX

       CCNOT
     - :python:`"CCX"`
     - .. math:: CCX = \begin{bmatrix}I & 0 & 0 & 0\\ 0 & I & 0 & 0 \\ 0 & 0 & I & 0 \\ 0 & 0 & 0 & X \end{bmatrix} = \begin{bmatrix}1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\  \end{bmatrix}
   * - Echoed cross-resonance gate
     - :python:`"ECR"`
     - .. math:: ECR = \begin{bmatrix}0 & V^\dagger \\ V & 0 \end{bmatrix} = \frac{1}{2}\begin{bmatrix}0 & 0 & 1 & i \\ 0 & 0 & i & 1 \\ 1 & -i & 0 & 0 \\ i & 1 & 0 & 0  \end{bmatrix}
   * - Swap gate
     - :python:`"SWAP"`
     - .. math:: SWAP = \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1  \end{bmatrix}
   * - Controlled swap gate
     - :python:`"CSWAP"`
     - .. math:: CSWAP =  \begin{bmatrix}I & 0 \\ 0 & SWAP \end{bmatrix} = \begin{bmatrix}1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1  \end{bmatrix}
   * - Rotation around X axis
     - :python:`"Rx"`
     - .. math:: R_X(\theta) = \exp\left(-i \frac{\pi}{2} \theta X\right) = \begin{bmatrix} \cos( \frac{\pi}{2} \theta) & - \sin( \frac{\pi}{2} \theta) \\ - \sin( \frac{\pi}{2} \theta) & \cos( \frac{\pi}{2} \theta) \end{bmatrix}
   * - Rotation around X axis
     - :python:`"Ry"`
     - .. math:: R_Y(\theta) = \exp\left(-i \frac{\pi}{2} \theta Y\right) = \begin{bmatrix} \cos( \frac{\pi}{2} \theta) & i \sin( \frac{\pi}{2} \theta) \\ - i \sin( \frac{\pi}{2} \theta) & \cos( \frac{\pi}{2} \theta) \end{bmatrix}
   * - Rotation around Z axis
     - :python:`"Rz"`
     - .. math:: R_Z(\theta) = \exp\left(-i \frac{\pi}{2} \theta Z\right) = \begin{bmatrix} \cos( \frac{\pi}{2} \theta) + \sin( \frac{\pi}{2} \theta) & 0 \\ 0 & \cos( \frac{\pi}{2} \theta) - \sin( \frac{\pi}{2} \theta) \end{bmatrix}
   * - Controlled rotation around X axis
     - :python:`"CRx"`
     - .. math:: CR_X(\theta) =  \begin{bmatrix}I & 0\\ 0 & RX(\theta) \end{bmatrix}  = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & \cos( \frac{\pi}{2} \theta) & - \sin( \frac{\pi}{2} \theta) \\ 0 & 0 & - \sin( \frac{\pi}{2} \theta) & \cos( \frac{\pi}{2} \theta) \end{bmatrix}
   * - Controlled rotation around Y axis
     - :python:`"CRy"`
     - .. math:: CR_Y(\theta) = \begin{bmatrix}I & 0\\ 0 & RY(\theta) \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & \cos( \frac{\pi}{2} \theta) & i \sin( \frac{\pi}{2} \theta) \\ 0 & 0 & - i \sin( \frac{\pi}{2} \theta) & \cos( \frac{\pi}{2} \theta) \end{bmatrix}
   * - Controlled rotation around Z axis
     - :python:`"CRz"`
     - .. math:: CR_Z(\theta) = \begin{bmatrix}I & 0\\ 0 & RZ(\theta)\end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & \cos( \frac{\pi}{2} \theta) + \sin( \frac{\pi}{2} \theta) & 0 \\ 0 & 0 & 0 & \cos( \frac{\pi}{2} \theta) - \sin( \frac{\pi}{2} \theta) \end{bmatrix}
   * - U3
     - :python:`"U3"`
     - .. math:: U3(\alpha,\beta,\gamma) = \exp((\alpha + \beta) i \frac{\pi}{2}) R_Z(\beta) R_Y(\alpha) R_Z(\gamma)
   * - U1
     - :python:`"U1"`
     - .. math:: U1(\gamma) = U3(0, 0, \gamma)
   * - U2
     - :python:`"U2"`
     - .. math:: U2(\beta, \gamma) = U3(0.5, \beta, \gamma)
   * - Controlled U3
     - :python:`"CU3"`
     - .. math:: CU3(\alpha,\beta,\gamma) = \begin{bmatrix}I & 0\\ 0 & U3(\alpha,\beta,\gamma)\end{bmatrix}
   * - Controlled U1
     - :python:`"CU1"`
     - .. math:: CU1(\gamma) = \begin{bmatrix}I & 0\\ 0 & U1(\gamma)\end{bmatrix}
   * - Controlled U2
     - :python:`"CU2"`
     - .. math:: CU2(\beta, \gamma) = \begin{bmatrix}I & 0\\ 0 & U2(\beta, \gamma)\end{bmatrix}
   * - Imaginary swap
     - :python:`"ISWAP"`
     - .. math:: iSWAP(\theta) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos( \frac{\pi}{2} \theta) & i \sin( \frac{\pi}{2} \theta) & 0 \\ 0 & i \sin( \frac{\pi}{2} \theta) & \cos( \frac{\pi}{2} \theta) & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
   * - Phased imaginary swap
     - :python:`"PhasedISWAP"`
     - .. math:: PhasedISWAP(\phi, \theta) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos( \frac{\pi}{2} \theta) & \exp(2i\pi \phi) i \sin( \frac{\pi}{2} \theta) & 0 \\ 0 & \exp(- 2i\pi \phi) i \sin( \frac{\pi}{2} \theta) & \cos( \frac{\pi}{2} \theta) & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
   * - XXPhase

       XX interaction
     - :python:`"XXPhase"`
     - .. math:: R_{XX}(\theta) = \exp\left(\frac{\pi}{2} \theta X\otimes X\right) = \begin{bmatrix} \cos( \frac{\pi}{2} \theta) & 0 & 0 & -i \sin( \frac{\pi}{2} \theta) \\ 0 & \cos( \frac{\pi}{2} \theta) & -i \sin( \frac{\pi}{2} \theta) & 0 \\ 0 & -i \sin( \frac{\pi}{2} \theta) & \cos( \frac{\pi}{2} \theta) & 0 \\ -i \sin( \frac{\pi}{2} \theta) & 0 & 0 & \cos( \frac{\pi}{2} \theta) \end{bmatrix}
   * - YYPhase

       YY interaction
     - :python:`"YYPhase"`
     - .. math:: R_{YY}(\theta) = \exp\left(\frac{\pi}{2} \theta Y\otimes Y\right) = \begin{bmatrix} \cos( \frac{\pi}{2} \theta) & 0 & 0 & i \sin( \frac{\pi}{2} \theta) \\ 0 & \cos( \frac{\pi}{2} \theta) & -i \sin( \frac{\pi}{2} \theta) & 0 \\ 0 & -i \sin( \frac{\pi}{2} \theta) & \cos( \frac{\pi}{2} \theta) & 0 \\ i \sin( \frac{\pi}{2} \theta) & 0 & 0 & \cos( \frac{\pi}{2} \theta) \end{bmatrix}
   * - ZZPhase

       ZZ interaction
     - :python:`"ZZPhase"`
     - .. math:: R_{ZZ}(\theta) = \exp\left(\frac{\pi}{2} \theta Z\otimes Z\right) = \begin{bmatrix} \exp( -i \frac{\pi}{2} \theta) & 0 & 0 & 0 \\ 0 & \exp( i \frac{\pi}{2} \theta) & 0 & 0 \\ 0 & 0 & \exp( i \frac{\pi}{2} \theta) & 0 \\ 0 & 0 & 0 & \exp( -i \frac{\pi}{2} \theta) \end{bmatrix}
   * - ZZMax
     - :python:`"ZZMax"`
     - .. math:: ZZMax = R_{ZZ}(0.5)
   * - PhasedX
     - :python:`"PhasedX"`
     - .. math:: PhasedX(\theta, \phi) = R_Z(\phi)R_X(\theta)R_Z(-\phi)
   * - TK1
     - :python:`"TK1"`
     - .. math:: TK1(\alpha,\beta,\gamma) = R_Z(\alpha)R_X(\beta)R_Z(\gamma)
   * - TK2
     - :python:`"TK2"`
     - .. math:: TK2(\alpha,\beta,\gamma) = R_{XX}(\alpha)R_{YY}(\beta)R_{ZZ}(\gamma)
