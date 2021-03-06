# Blur-estimation
Реализовано два метода: один - классическим приемом компьютерного зрения (КЗ), другой - с помощью сверточной нейронной сети (СНС). 
Метод классического КЗ основывается на вычислении дисперсии после применения ФВЧ и был позаимствован отсюда: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/ с незначительными изменениями.

СНС была взята за основу отсюда: https://github.com/priyabagaria/Image-Blur-Detection. Сеть была немного изменена: добавлено больше слоев и изменены некоторые параметры. Набор тренировочных данных взят за основу тот же, что и по приведенной ссылке, однако были добавлены некоторые изображения с этого набора: http://vllab.ucmerced.edu/wlai24/cvpr16_deblur_study/, с этого: http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/index.html#downloads, а также собственные фотографии. Тестовый набор оставлен без изменений. Итоговые наборы при желании можно скачать отсюда: https://drive.google.com/drive/folders/1aR5vGxhesUE_npwKueehzr6ebOkvG5eX. 
Полную итоговую модель в формате .h5 можно взять здесь: https://drive.google.com/file/d/1MTYttJY8OuM1KOsDfxPVMc0t-2YfBDy7/view?usp=sharing

Сравнение и анализ полученных результатов (natural set - набор четких изображений и изображений с непроизвольным размытием, т.е. без искусственного применения ФНЧ; digital set - набор четких изображений и изображений с искусственным размытием, т.е. с применением ФНЧ):

CV:

CV accuracy on a natural set =  0.764

CV accuracy on a digital set =  0.970

Total accuracy = 0.831

Precision =  0.830

Recall =  0.881

F-measure =  0.855

MCC =  0.640


CNN:

CNN accuracy on a natural set =  0.751

CNN accuracy on a digital set =  0.846

Total accuracy = 0.781

Precision =  0.840

Recall =  0.771

F-measure =  0.804

MCC =  0.561

Как видно, оба метода имеют достаточно хорошую долю правильных ответов при тестировании на наборе с четкими и искусственно размытыми изображениями (при этом классический метод выигрывает на 13%) и меньшую точность на наборе с четкими и натурально размытыми изображениями (около 75%). Возможно, это объясняется тем, что на искусственно размытых изображениях применялся только один фильтр, в то время как натурально размытые изображения могут быть представлены совокупностью различных фильтров и причин, вызвавших нечеткость. Итоговая доля правильных ответов получилась чуть выше у классического метода (83% против 78%). Точность (precision) у сети немного выше (84% против 83%). Однако полнота (recall) у классического метода заметно выше (88% против 77%). По итогу корреляционный коэффициент Мэтьюза (MCC) оказался выше у классического метода (0.64 против 0.56).

Однако данный классический метод имеет и ряд недостатков: например, натуральные природные объекты без ярко выраженных границ алгоритм также относит к классу blurred (примеры: облака, туман и т.д.). То же справедливо и для изображений с отсутствием контуров в принципе (пример: синее безоблачное небо). Нейронная сеть в каждом отдельном случае может отнести подобные объекты в разные категории. 

Оба алгоритма инвариантны к повороту изображения (для резких и размытых изображений дисперсия (для классического метода) и вероятности принадлежности к классу (для нейронной сети) остаются теми же. Были протестированы углы поворота 0, 90, 180, 270 градусов).

Предварительно сохраненная и скомпилированная модель сети выдает результат предсказаний на тестовом наборе (1480 изображений) в среднем за 3 секунды. Классический метод тратит примерно 0.3с только лишь на одно изображение, однако это связано с тем, что мой способ обхода по файлам далеко не самый оптимальный; при правильном подходе это время можно сократить. Если добавить время обучения модели к времени предсказания, то общее время, затрачиваемое машиной при использовании нейронной сети и классического метода, сопоставимо.

Порог для классического метода был подобран эмпирически, исходя из собственного представления о размытости изображения, при его смене результаты работы алгоритма также изменятся.

Здесь также не были рассмотрены случае, когда часть изображения явно выражена, а некоторые регионы - заблюрены, в т.ч. и специально. Все-таки это отдельная тема (например: http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/index.html#downloads - исследование на эту тему).

Итого: в целом в данном случае классический метод немного выигрывает у нейронной сети, однако, возможно, что при усложнении сети и увеличении размера тренировочной выборки можно добиться гораздо более аккуратных результатов. Метод классического компьютерного зрения хорош простотой реализации при неплохих получаемых результатах, однако и сильно зависит от выбранного порога срабатывания.  
