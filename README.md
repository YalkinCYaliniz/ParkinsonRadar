# Takım İsmi  
AI 121

# PARKINSONRADAR

Konuşma analizi ile Parkinson hastalığının erken evre tespitine odaklanan bir yapay zeka destekli sağlık teknolojisi projesidir.  
Kullanıcıların yalnızca birkaç cümlelik konuşma kaydı yüklemesiyle çalışan sistem, ses işleme ve makine öğrenmesi tekniklerini kullanarak Parkinson riskini belirler.  
Amaç, bireylerde hastalığın henüz klinik semptomlar ortaya çıkmadan yakalanabilmesini sağlamak ve uzmanlara erken müdahale fırsatı sunmaktır.

## Takım Üyeleri  
- Product Owner: Yalkın Can Yalınız  
- Scrum Master: Birgül Taşdemir 
- Developer: Ayhan Gurbangeldiyev  
- Developer: Tunahan Aydın 

## Ürün İsmi  
ParkinsonRadar

## Ürün Açıklaması  

ParkinsonRadar, bireylerin kısa konuşma kayıtlarını analiz ederek Parkinson hastalığına dair erken dönem risk sinyallerini tespit etmeye odaklanan bir sağlık uygulamasıdır.  
Kullanıcılar sisteme konuşma seslerini yükler; sistem, bu sesleri analiz eder ve bir risk puanı sunar. Bu analiz, doğal dil işleme ve ses işleme teknikleriyle desteklenmiş bir makine öğrenimi modeli ile gerçekleştirilir.  

## Ürün Özellikleri  
- Kullanıcının kısa bir konuşma kaydını yükleyebilmesi  
- Yüklenen sesin doğal dil işleme ve ses işleme ile analiz edilmesi  
- Parkinson riskinin belirlenmesi ve puan olarak sunulması  
- Risk değerlendirmesine etki eden ses özelliklerinin açıklanması  
- Geçmiş analiz sonuçlarının saklanması ve karşılaştırmalı görüntülenmesi  
- Doktorlar için detaylı analiz sunan uzman modu  
- Anonimleştirilmiş ve veri gizliliği ilkesine uygun çalışma  
- Web tabanlı kullanıcı arayüzü ile kolay erişim  
- Sonuçların e-posta ile raporlanabilmesi  
- Simüle edilmiş veri setleriyle test edilmiş ve doğrulanmış model

## Hedef Kitle  
- 45 yaş üstü bireyler  
- Aile hekimleri ve nörologlar  
- Parkinson riskini öğrenmek isteyen genel kullanıcılar  
- Klinik araştırmacılar

## Product Backlog  
[Miro Backlog Board](https://miro.com/app/board/uXjVIgjwiGI=/?share_link_id=736731013650)
## SPRINT 1
- **Sprint Notları** : Takım rolleri belirlendi. Ürün belirlendi. Kaynak araştırması yapıldı. Projede yapılacaklar işlemler backlog olarak agile boarda eklendi.  Veri setinin özellikleri incelendi ve kategorilere ayrıldı. Özellik isimleri sınıflandırıldı. Gerekli kütüphaneler belirlendi ve kurulumu yapıldı.
- **Tahmin Edilen/Tamamlanacak Puan:** Proje başında tamamlanması belirlenen tasklar tamamlanarak belirlenen 33 puan tamamlanmıştır. Sprint 1 için tahmin edilen ve hedeflenen puan 33'tür. Belirlenen tasklar tamamlanmış ve 33 puanlık iş yapılmıştır.
- ![image](https://github.com/user-attachments/assets/ab81bf0d-82d0-40a1-b7ff-1374a75ecb8d)
- **Tahmin Mantığı:** Yapılacak taskların her birine zorluk derecesi, yapılabilme süreleri ve önem derecelerine göre puanlandırma yapılmıştır ve toplam puan 130'dur. Sprint 1de daha düşük tasklar yapılmasına kararlaştırılmış ve diğer sprintlere nazaran daha az puan tamamlanmıştır. ilk sprint için tamamlanan toplam puan 33tür.
- **Daily Scrum** : Whatsapp üzerinden toplantılar organize edilmiştir. Yapılacak ürün belirlendi. Projedeki takım rolleri belirlenmiştir. Veri kaynağı araştırmaları yapılmıştır.
- ![image](https://github.com/user-attachments/assets/e4ebd862-2bd8-4b19-8638-343de9d03b4e)
- ![image](https://github.com/user-attachments/assets/c19dac36-6f26-4d0c-973b-536304cfa3ab)
- **Sprint Board Updates**
- ![Agile Board](https://github.com/user-attachments/assets/73c6ab46-4aa8-4ac2-b1d0-292cc4cb4d86)
- **Screenshot**
![image](https://github.com/user-attachments/assets/68b63a19-5ae8-4f6b-8989-89dc1cdb9d36)
![image](https://github.com/user-attachments/assets/ba3c9158-5b5a-4723-b1c2-c33c31bcbca9)
![image](https://github.com/user-attachments/assets/faafdb4e-68bb-4b55-8c16-6e7d1d805d3a)



- **Sprint Review** : Alınan kararlar: Hangi veri setinin kullanılacağı belirlendi. 
- **Sprint Retrospective** : Diğer 2 sprintte daha verimli çalışılacağına ve daha düzenli toplantılar yapılması kararlaştırıldı.

## SPRINT 2
- **Daily Scrum** : Toplantılar zoom üzerinden gerçekleşmiştir.
- **Sprint Review** :  Bu sprint'te projemizde başlangıçta 755 sütunlu karmaşık veri setinden computational efficiency ve klinik doğruluk için 22 sütunlu UCI Parkinson's Dataset'e geçiş yaparak optimizasyon sağladık, ardından Jitter, Shimmer, NHR, RPDE, DFA, PPE gibi ses özelliklerini Parselmouth ve Librosa kullanarak çıkaran comprehensive bir pipeline geliştirdik ve her modeli farklı amaçlar için özelleştirdiğimiz ensemble model sistemi kurarak Flask tabanlı real-time audio recording, drag & drop file upload, interactive Plotly visualizations içeren responsive web uygulaması oluşturduk - Random Forest'ı hangi ses özelliğinin en önemli olduğunu anlamak için, XGBoost'u yüksek doğruluk oranı elde etmek için, LightGBM'i hızlı sonuç almak için, SVM'i karmaşık ses pattern'lerini ayırt etmek için, Neural Networks'ü ise insan kulağının bile fark edemeyeceği ince detayları yakalamak için kullandık, kritik olan mikrofon kayıtlarının dataset'le uyumsuzluğunu (consumer mikrofon ~45% jitter vs profesyonel stüdyo dataset ~0.6% jitter) multi-layer normalization stratejisi ile çözdük - önce statistical outlier detection (Z-score >3.0 için dataset mean+2σ correction), sonra physiological range clamping (jitter max %5, shimmer max %15), ardından ultra-strict validation (hala yüksekse healthy average'a çekme) ve son olarak adaptive correction factors ile environment detection yaparak noisy_environment için %95 jitter reduction, clean_environment için %70 reduction uygulayarak gerçekçi değerlere normalize ettik, normalize edilmiş küçük değerlerin (0.006 jitter) grafiklerde görünmez olması problemini feature-specific multiplier sistemi (×2000'e kadar çarpanlar) ve detailed hover tooltips ile çözdük, görselleştirme sisteminde kullanıcıların her bir ses özelliğini tek tek inceleyebilmesi, mouse ile grafik üzerinde gezinerek detaylı değerleri görebilmesi, analiz sonuçlarını JSON formatında export edebilmesi ve comprehensive PDF rapor olarak indirebilmesi özelliklerini ekledik.
- **Tahmin Edilen/Tamamlanacak Puan:** Sprint 2 için tahmin edilen ve hedeflenen puan 53'tür. Belirlenen tasklar tamamlanmış ve 53 puanlık iş yapılmıştır.
ÖZELLIK           ÖNCESI    SONRASI    İYİLEŞME
Jitter            45.2%  →  0.0062%    %99.9 azalma
Shimmer           12.8%  →  0.0234%    %99.8 azalma  
NHR               0.87   →  0.095      %89 azalma
RPDE              0.083  →  0.387      %366 artış (doğru yön)
Model Accuracy    12%    →  87%       %625 iyileşme

- ![image] (https://github.com/user-attachments/assets/71469302-1e34-4d39-818d-03789a74d2a5)
- ![image] (https://github.com/user-attachments/assets/75968a54-c9b2-46be-b31e-801487d0d744)
- ![image] (https://github.com/user-attachments/assets/6727bd9b-95ec-429f-8eba-991a6c81c651)
- ![image] (https://github.com/user-attachments/assets/60b0ff3e-da6c-4bc1-a424-74ae5ec9c2c5)
- ![image] (https://github.com/user-attachments/assets/eca69f9e-0683-4178-a147-1d44a1a6d7ee)
- ![image] (https://github.com/user-attachments/assets/17529d48-48b6-40b9-8467-e7356cc5d249)
- ![image] (https://github.com/user-attachments/assets/792c78f1-6d07-48d3-9895-d19ba91e529a)
- ![image] (https://github.com/user-attachments/assets/74df66e6-6c2a-497d-87e1-1994b3810423)

- **Sprint Retrospective** : Sprint 2'de veri seti optimizasyonu, ensemble model kurulumu ve Flask tabanlı web uygulaması geliştirme ile önemli ilerleme kaydettik.Ancak bu sprint'te karşılaştığımız temel zorluklardan biri, veri setinin kaydedildiği profesyonel stüdyo mikrofonları ile bizim kullandığımız tüketici mikrofonları arasındaki uyumsuzluktan kaynaklanan çevresel faktörler ve ses kalitesi farklılıklarıydı. Özellikle jitter gibi ses özelliklerinde ciddi sapmalar yaşadık. Bu sorunu, çok katmanlı normalizasyon stratejisi (istatistiksel aykırı değer tespiti, fizyolojik aralık sınırlaması, ultra-katı doğrulama ve adaptif düzeltme faktörleri) uygulayarak başarılı bir şekilde giderdik. Bu sayede, farklı kayıt ortamlarından gelen verileri başarılı bir şekilde standardize edebildik.



  
