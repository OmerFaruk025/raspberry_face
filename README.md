# Pi-FaceID

Pi-FaceID, Raspberry Pi veya bilgisayar üzerinde çalışan gerçek zamanlı yüz tanıma sistemidir.
Proje; yüz verisi toplama, model eğitimi ve canlı tanıma adımlarından oluşan uçtan uca bir FaceID çözümü sunar.

Sistem kamera üzerinden yüz algılar, tanıma işlemini gerçekleştirir ve tanınan kişileri zaman bilgisiyle birlikte log dosyasına kaydeder.

---

## Özellikler

- Gerçek zamanlı yüz algılama ve tanıma
- Hızlı ve stabil yüz veri toplama
- LBPH tabanlı yüz tanıma modeli
- Tanınan kişilerin CSV formatında loglanması
- CLI tabanlı sade kullanım
- Raspberry Pi ve PC uyumlu yapı

---

## Çalışma Akışı

1. Kamera üzerinden yüz verileri toplanır(collect_faces.py) 
2. Toplanan yüzler ile model eğitilir(train_lbph.py)
3. Canlı kamera akışında yüz tanıma yapılır(recognize.py)
4. Tanınan kişiler log dosyasına kaydedilir(recognize.py)

---

## Kullanım

### Yüz Verisi Toplama

1-) Yüz kaydı almak için öncelikle kişinin tanıtılması gerekmektedir. "collect_faces.py" dosyası çalıştırıldığında sistem belli miktarda fotoğrafları kaydeder.
- Bu işlem yapılırken kaç tane fotoğrafın kaydedildiğini görebilirsiniz.
- Fotoğrafların konumunu şurada olacaktır: data/faces/... isimli şahıs
- BU İŞLEMİ HER SEFERİNDE GERÇEKLEŞTİRMENİZE GEREK YOK. YENİ KİŞİYİ TANITIRKEN VEYA GÜNCELLERKEN ÇALIŞTIRMANIZ YETERLİ.
---------

2-) Kaydedilen fotoğrafları taratmak gerekiyor. Bunun için "train_lbph.py" dosyası çalıştırılmalıdır.
- Ekstra yapmanız gereken bir işlem yok. bu dosya ana dosyanın kök dizisine .yml dosyası ve label.txt dosyası oluşturacaktır.
- Bu dosya data klasöründeki tüm kişileri işleyip sadece sağlıklı fotoğrafları tarayacaktır.
- BU İŞLEMİ HER SEFERİNDE YAPMANIZA GEREK YOK. YENİ KİŞİ KAYDETTİKTEN SONRA VEYA GÜNCELLEDİKTEN SONRA ÇALIŞTIRMANIZ GEREKİR AKSİ TAKDİRDE ESKİ FOTOĞRAFLAR BAZ ALINIR.
---------

3-) Önceki iki işlemi yapmayı başardıysanız eğer artık "recognize.py" dosyasını çalıştırarak anlık olarak Raspberry kamerası üzerinden tanıma yapabilirsiniz.
- Bu dosya tanımış olduğu kişiyi .csv dosyasına kaydeder. Bu sayede kimlerin yüzünü tanıttığına dair detaylı bilgi sahibi olabilirsiniz.
- Eğer uzaktan başlatmak veya kontrol etmek istiyorsanız lütfen bu dosya yerine 4. aşamadaki .py dosyasını çalıştırın.
---------

4-) Uygulamayı dışarıdan kontrol etmek isterseniz ve kamerayı canlı olarak görmek isterseniz "web_controller.py" dosyasını çalıştırmanız gerekmekte.
- Bu dosya dolaylı yoldan recognize.py dosyasını çalıştırdığı için tekrardan o dosyayı çalıştırmanıza gerek yok.
- Bu arayüzü kullanabilmek için Raspberry ile aynı internete bağlı olmanız gerekiyor
- !!!!!!!!! "web_controller.py" link vermezse veya verdiği link çalışmazsa manuel olarak kendiniz url yazmanız gerekmektedir.!!!!!!!!
- !!!!!!!!!Modem arayüzünden Pi nin ip adresini öğrenerek url kısmına şunu yazmanız gerekiyor. "ip_adresiniz:5000"!!!!!!!!
- Bu sayede artık web arayüzüne girebiliyor olmalısınız.
---------

## License
This project is not open-source. All rights reserved.
