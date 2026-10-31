"""A hard, style-matched set of calls, SMS and emails, written for this project.

WHY: in the public data, scam and safe texts differ in obvious ways (and the safe calls are written
dialogues), so models score near 100 %. Here the *safe* texts deliberately use scam vocabulary
("verify", "code", "account", "refund", "press 1", "gift card", "urgent") and the *fraud* texts are
often polite, calm and keyword-free. Both classes are assembled by the same procedure from the same
kind of script skeleton with similar length, so writing style does not give the answer away.
Scenarios are Puerto Rico flavoured (Banco Popular, FirstBank, AutoExpreso, ATH Móvil, LUMA, Hacienda,
DACO ...) in English and Spanish.

HONEST LIMITS - read before quoting results:
  * These are synthetic reconstructions written from scam patterns that are public (DACO / FTC alerts),
    NOT real recorded cases. Labels are correct by construction (the author knows which is which).
  * All of them come from a small number of hand-written skeletons. Every skeleton is one *group*
    that is never split between train and test, so scores on this set measure generalisation to
    scripts the model has not seen, but they still cannot stand in for real DACO cases.
  * No real people, numbers, accounts or live links: links use made-up, non-existent domains for fraud and
    well-known public domains for safe messages.

Each skeleton is a list of steps; each step is a list of alternative sentences. A variant picks one
alternative per step and fills the {slots}.
"""
from __future__ import annotations

import random
import re

import pandas as pd

from . import config as C

NAMES_EN = ["Karen", "Michael", "Jessica", "Daniel", "Laura", "Robert", "Maria", "Steven", "Angela", "Brian"]
NAMES_ES = ["Carmen", "José", "Luis", "Ana", "Rafael", "Yolanda", "Héctor", "Marisol", "Pedro", "Lourdes"]
CITIES = ["San Juan", "Bayamón", "Ponce", "Caguas", "Mayagüez", "Carolina", "Arecibo", "Guaynabo", "Humacao", "Aguadilla"]
BANKS = ["Banco Popular", "FirstBank", "Oriental Bank"]
AMOUNTS = ["$49.99", "$127.50", "$310.00", "$86.25", "$212.40", "$18.75", "$455.10", "$63.00"]
TIMES = ["tomorrow at 9:30 AM", "Thursday at 2:00 PM", "Monday at 11:15 AM", "Friday at 4:30 PM", "next Tuesday at 10:00 AM"]
TIMES_ES = ["mañana a las 9:30 AM", "el jueves a las 2:00 PM", "el lunes a las 11:15 AM", "el viernes a las 4:30 PM", "el martes a las 10:00 AM"]
FAKE_LINKS = ["https://popular-verificar-cuenta.top/acceso", "https://autoexpreso-pagos.click/saldo", "https://ath-movil-soporte.xyz/login",
              "https://luma-factura-pr.online/pago", "https://usps-redelivery-fee.support/track", "https://secure-firstbank-alerta.cc/verify",
              "https://amazon-order-hold.work/confirm", "https://hacienda-reembolso-pr.top/form", "https://oriental-bank-aviso.site/acceso",
              "https://microsoft-account-review.link/signin", "https://bit.ly/3Hk9Pq2", "https://tinyurl.com/yc5k8vwa"]
REAL_LINKS = ["https://www.popular.com/en/", "https://www.firstbankpr.com/", "https://www.autoexpreso.com/", "https://www.ups.com/track",
              "https://www.usps.com/", "https://www.pr.gov/", "https://www.walgreens.com/", "https://www.cvs.com/", "https://www.fedex.com/en-us/tracking.html",
              "https://www.irs.gov/refunds", "https://hacienda.pr.gov/", "https://www.oriental.com/"]


SCENARIO_LINKS = [("toll", ["https://autoexpreso-pagos.click/saldo", "https://autoexpreso-peaje.top/pago"]),
                  ("bank", ["https://popular-verificar-cuenta.top/acceso", "https://secure-firstbank-alerta.cc/verify", "https://oriental-bank-aviso.site/acceso"]),
                  ("ath", ["https://ath-movil-soporte.xyz/login", "https://athmovil-validar.online/cuenta"]),
                  ("luma", ["https://luma-factura-pr.online/pago", "https://luma-pago-seguro.click/factura"]),
                  ("usps", ["https://usps-redelivery-fee.support/track"]), ("package", ["https://usps-redelivery-fee.support/track", "https://bit.ly/3Hk9Pq2"]),
                  ("amazon", ["https://amazon-order-hold.work/confirm"]), ("prize", ["https://hacienda-reembolso-pr.top/form", "https://tinyurl.com/yc5k8vwa"]),
                  ("hacienda", ["https://hacienda-reembolso-pr.top/form"]),
                  ("password", ["https://microsoft-account-review.link/signin"]), ("share", ["https://microsoft-account-review.link/signin"]),
                  ("invoice", ["https://microsoft-account-review.link/signin", "https://bit.ly/3Hk9Pq2"])]
SCENARIO_REAL = [("toll", ["https://www.autoexpreso.com/"]), ("bank", ["https://www.popular.com/en/", "https://www.firstbankpr.com/", "https://www.oriental.com/"]),
                 ("delivery", ["https://www.ups.com/track", "https://www.fedex.com/en-us/tracking.html", "https://www.usps.com/"]),
                 ("pharmacy", ["https://www.walgreens.com/", "https://www.cvs.com/"]), ("order", ["https://www.ups.com/track"]),
                 ("statement", ["https://www.popular.com/en/", "https://www.firstbankpr.com/"])]


def _scenario_link(sid: str, table, default, rng: random.Random) -> str:
    for key, links in table:
        if key in sid:
            return rng.choice(links)
    return rng.choice(default)


def _slots(rng: random.Random, lang: str) -> dict:
    return {"agent": rng.choice(NAMES_ES if lang == "es" else NAMES_EN), "name": rng.choice(NAMES_ES if lang == "es" else NAMES_EN),
            "bank": rng.choice(BANKS), "city": rng.choice(CITIES), "amount": rng.choice(AMOUNTS), "last4": f"{rng.randint(1000, 9999)}",
            "time": rng.choice(TIMES_ES if lang == "es" else TIMES), "n": rng.randint(2, 9), "code": f"{rng.randint(100000, 999999)}",
            "fake": rng.choice(FAKE_LINKS), "real": rng.choice(REAL_LINKS)}


# ===================================================================== CALLS
# (skeleton id, language, steps)
FRAUD_CALLS = [
    ("bank_en", "en", [
        ["Hello, this is {agent} from the fraud prevention team at {bank}.", "Good afternoon, my name is {agent} and I'm calling from {bank} security.", "Hi, this is {agent} with {bank}'s account protection department."],
        ["We noticed a purchase of {amount} in {city} on your debit card ending in {last4} and wanted to make sure it was you.", "Our system flagged a payment of {amount} at an online store that doesn't match your usual activity.", "There was a transfer of {amount} attempted from your checking account this morning."],
        ["Since you didn't make it, I can block it right now and send you a replacement card.", "I can reverse it today, it only takes a couple of minutes.", "No worries, I can cancel it and protect your balance while we're on the phone."],
        ["To confirm it's really you, I'm sending a six digit code to your phone, could you read it back to me when it arrives?", "For verification I just need the code we text you and the last four digits of your social security number.", "Please read me the code on your screen and the online banking password you use, so I can lock the old one."],
        ["Thank you for your patience, this will only take a moment.", "You're doing the right thing by acting quickly, our team will keep your funds safe.", "Once I have that, I'll move your funds to a protected account until the investigation closes."]]),
    ("bank_es", "es", [
        ["Buenas tardes, le habla {agent} del departamento de prevención de fraude de {bank}.", "Hola, mi nombre es {agent} y llamo de seguridad de {bank}.", "Saludos, soy {agent} de la unidad de protección de cuentas de {bank}."],
        ["Detectamos una compra de {amount} en {city} con su tarjeta terminada en {last4} y queremos confirmar que fue usted.", "Nuestro sistema marcó un pago de {amount} que no corresponde a su actividad normal.", "Se intentó una transferencia de {amount} desde su cuenta de cheques esta mañana."],
        ["Si usted no la hizo, puedo bloquearla ahora mismo y enviarle una tarjeta nueva.", "Puedo revertirla hoy mismo, solo toma un par de minutos.", "No se preocupe, la cancelo y protejo su balance mientras hablamos."],
        ["Para confirmar que es usted le estoy enviando un código de seis dígitos por mensaje de texto, ¿me lo puede leer cuando le llegue?", "Para verificar solo necesito el código que le enviamos y los últimos cuatro dígitos de su seguro social.", "Por favor léame el código de la pantalla y la contraseña de su banca en línea para bloquear la anterior."],
        ["Gracias por su paciencia, esto será rápido.", "Hace bien en actuar rápido, nuestro equipo cuidará sus fondos.", "Cuando tenga eso, moveré sus fondos a una cuenta protegida hasta que cierre la investigación."]]),
    ("toll_es", "es", [
        ["Buenos días, le llamamos de AutoExpreso, mi nombre es {agent}.", "Hola, habla {agent} del servicio al cliente de AutoExpreso.", "Buenas tardes, soy {agent} del departamento de cobros de AutoExpreso."],
        ["Su cuenta del peaje tiene un balance pendiente de {amount} de los últimos meses.", "Nos aparece que usted tiene {n} cruces sin pagar en su tablilla.", "Su tablilla quedó desactivada por un pago que no se pudo procesar."],
        ["Si lo resuelve hoy evitamos el cargo por mora y la multa del Departamento de Transportación.", "Podemos arreglarlo por teléfono ahora mismo para que no le llegue la multa a su casa.", "Es un trámite sencillo y le ahorra la penalidad."],
        ["Necesito el número de su tarjeta de crédito con la fecha de vencimiento para aplicar el pago.", "Le puedo enviar un enlace por mensaje de texto para que pague con su tarjeta, o me da los datos aquí.", "Puede pagar con tarjetas de regalo prepagadas si no tiene tarjeta, yo le indico cómo."],
        ["Con eso queda todo saldado, le llegará un mensaje de confirmación.", "Gracias por su cooperación, que tenga un buen día.", "No cuelgue hasta que le confirme que el pago entró."]]),
    ("toll_en", "en", [
        ["Good morning, this is {agent} from AutoExpreso billing.", "Hello, my name is {agent} and I'm calling about your AutoExpreso toll account.", "Hi, {agent} here with the AutoExpreso collections office."],
        ["Our records show an unpaid balance of {amount} on your toll tag.", "You have {n} unpaid toll crossings that were never charged to your card.", "Your tag was suspended after a payment failed to process."],
        ["If we settle it today we can waive the late fee and the Transportation Department fine.", "We can fix this over the phone right now so a penalty notice doesn't reach your home.", "It's a quick process and it saves you the penalty."],
        ["I'll need your credit card number and expiration date to apply the payment.", "I can text you a payment link, or you can read me the card details here.", "If you don't have a card you can pay with prepaid gift cards and I'll walk you through it."],
        ["That settles everything and you'll get a confirmation text.", "Thank you for cooperating, have a great day.", "Please stay on the line until I confirm the payment went through."]]),
    ("athmovil_es", "es", [
        ["Hola, le habla {agent} del soporte de ATH Móvil.", "Buenas tardes, soy {agent} del equipo de seguridad de ATH Móvil.", "Saludos, mi nombre es {agent}, llamo de parte de ATH Móvil."],
        ["Vimos un intento de acceso a su cuenta desde otro teléfono en {city}.", "Su cuenta de ATH Móvil va a quedar limitada porque no se ha validado la identidad.", "Se registró un envío de {amount} a un número que usted no tiene guardado."],
        ["Para detenerlo necesito que me ayude con un paso rápido desde su aplicación.", "Puedo cancelar ese envío si me confirma unos datos.", "Esto es parte de nuestro proceso de protección, no le tomará más de unos minutos."],
        ["Entre a la aplicación y envíe {amount} a la cuenta de seguridad que le voy a dar, se la devolvemos enseguida.", "Dígame el código de verificación que le llegó y el PIN de su ATH Móvil.", "Instale la aplicación de asistencia remota que le indico para revisar su teléfono."],
        ["Gracias, con eso su cuenta queda protegida.", "Perfecto, ya casi terminamos, no cuelgue.", "Le agradezco que haya colaborado, cualquier cosa nos llama de nuevo."]]),
    ("techsupport_en", "en", [
        ["Hello, I'm {agent} from the Windows technical support center.", "Good afternoon, this is {agent} calling from Microsoft customer care.", "Hi, my name is {agent}, I'm with the PC security division."],
        ["We've been receiving error reports from your computer for the past few days.", "Your router and computer are sending out a signal that shows they may be infected.", "Our servers detected suspicious activity connected to your internet account."],
        ["It's nothing to worry about, we fix this for customers every day.", "The good news is we can clean it remotely in about ten minutes.", "I'll guide you step by step so you don't have to do anything complicated."],
        ["Please open your computer and install the support program I'm about to read you the address for.", "I need you to let me connect to your screen and then log in to your bank so I can check nothing was taken.", "There is a one-time service fee of {amount}, which you can pay with gift cards from any pharmacy."],
        ["Thank you for working with me, your computer will be much safer.", "Once the program is installed you can relax, I'll take care of everything.", "Please don't hang up until the scan is finished."]]),
    ("techsupport_es", "es", [
        ["Buenas, le habla {agent} del centro de soporte técnico de Windows.", "Buenos días, soy {agent} de servicio al cliente de Microsoft.", "Hola, mi nombre es {agent}, trabajo en la división de seguridad de computadoras."],
        ["Hemos recibido reportes de errores de su computadora en los últimos días.", "Su computadora y su conexión de internet están enviando una señal de posible virus.", "Nuestros servidores detectaron actividad sospechosa ligada a su cuenta de internet."],
        ["No se preocupe, esto lo arreglamos todos los días con otros clientes.", "Lo bueno es que podemos limpiarla de forma remota en unos diez minutos.", "Yo le voy guiando paso a paso para que no tenga que hacer nada complicado."],
        ["Por favor encienda su computadora e instale el programa de soporte que le voy a dictar.", "Necesito que me deje conectarme a su pantalla y que entre a su banco para verificar que no se llevaron nada.", "Hay un cargo único de servicio de {amount} que puede pagar con tarjetas de regalo de cualquier farmacia."],
        ["Gracias por su ayuda, su computadora quedará mucho más segura.", "Cuando termine la instalación usted se relaja, yo me encargo de todo.", "Por favor no cuelgue hasta que termine el análisis."]]),
    ("benefits_en", "en", [
        ["Hello, this is {agent} calling from the Social Security benefits office.", "Good morning, my name is {agent}, I'm with the Internal Revenue Service refund unit.", "Hi, {agent} here from the federal benefits verification department."],
        ["Your file shows a refund of {amount} that hasn't been released because of a missing detail.", "We found a problem with the social security number linked to your benefits.", "There is a pending adjustment on your account that needs a quick confirmation."],
        ["This is routine and it's normal that you haven't heard from us by mail.", "I can release it today if we complete the verification now.", "I understand it sounds unusual, but this is how we handle it for seniors."],
        ["I only need to confirm your full social security number and your bank account number so I can deposit it.", "Please give me your date of birth, your SSN and the routing number of your bank.", "To reactivate it you'll pay a small processing fee with a prepaid card and then the money is released."],
        ["Thank you, the deposit should appear in two or three business days.", "You've been very helpful, I'll mark your file as complete.", "Please keep this call confidential until the deposit arrives."]]),
    ("benefits_es", "es", [
        ["Buenos días, le llama {agent} del Departamento de Hacienda, unidad de reembolsos.", "Saludos, soy {agent} de la oficina de verificación de beneficios del Seguro Social.", "Buenas tardes, mi nombre es {agent}, llamo de la división de reintegros."],
        ["Su expediente muestra un reembolso de {amount} que no se ha liberado por un dato incompleto.", "Encontramos un problema con el número de seguro social ligado a sus beneficios.", "Hay un ajuste pendiente en su cuenta que necesita una confirmación rápida."],
        ["Esto es un trámite de rutina y es normal que no haya recibido nada por correo.", "Puedo liberarlo hoy mismo si completamos la verificación ahora.", "Entiendo que suena raro, pero así lo hacemos con las personas de edad avanzada."],
        ["Solo necesito confirmar su número completo de seguro social y el número de su cuenta bancaria para depositarle.", "Deme por favor su fecha de nacimiento, su seguro social y el número de ruta de su banco.", "Para reactivarlo paga un pequeño cargo de procesamiento con una tarjeta prepagada y se libera el dinero."],
        ["Gracias, el depósito debe aparecer en dos o tres días laborables.", "Ha sido de mucha ayuda, voy a marcar su expediente como completo.", "Le pido que mantenga esta llamada en confidencia hasta que llegue el depósito."]]),
    ("package_en", "en", [
        ["Hi, this is {agent} from the delivery services department.", "Hello, I'm {agent} calling about a parcel addressed to you.", "Good afternoon, {agent} with the shipping customer center."],
        ["We tried to deliver a package to your address in {city} but nobody was available.", "Your package is on hold at our facility because the address couldn't be confirmed.", "A parcel with your name is being held and will be returned unless it's claimed."],
        ["It's just a small redelivery fee of {amount} and we'll bring it tomorrow.", "We can reschedule today if we settle a small customs charge.", "It happens a lot, it's quick to solve."],
        ["I'll text you a link so you can enter your card details and your address.", "Please read me your card number so I can apply the fee for the redelivery.", "Confirm your full name, date of birth and the code that you receive by text message."],
        ["Perfect, your package will arrive within two days.", "Thank you for your time, you'll receive a tracking number shortly.", "Have a wonderful day and thanks for choosing us."]]),
    ("utility_es", "es", [
        ["Buenos días, le habla {agent} del departamento de cobros de LUMA Energy.", "Saludos, mi nombre es {agent}, llamo de la compañía de energía eléctrica.", "Hola, soy {agent}, del área de facturación de LUMA."],
        ["Su cuenta aparece con una factura vencida de {amount}.", "Nos consta que dos pagos suyos fueron devueltos por el banco.", "Su servicio está programado para ser suspendido hoy por falta de pago."],
        ["Podemos evitarlo si hace el pago ahora mismo por teléfono.", "Un técnico va para su casa dentro de una hora a cortar el servicio, pero podemos detenerlo.", "Tenemos una opción de pago inmediato para que no pierda la luz."],
        ["Vaya a la farmacia, compre tarjetas de regalo por ese monto y léame los números.", "Necesito el número de su tarjeta de débito y el código de seguridad del dorso.", "Le envío un enlace para que pague de inmediato con su cuenta del banco."],
        ["Una vez recibido el pago le restablecemos todo de inmediato.", "Gracias, esto evita una reconexión que cuesta más de cien dólares.", "No cuelgue, necesito validar el pago en el sistema."]]),
    ("family_es", "es", [
        ["Hola abuela, soy yo, ¿me reconoces?", "Hola abuelo, soy tu nieto, por favor no se asuste.", "Buenas, tía, soy yo, necesito tu ayuda."],
        ["Tuve un problema, un accidente de carro en {city}, y estoy con un abogado.", "Me detuvieron por un error y necesito una fianza de {amount}.", "Se me perdió el celular y estoy usando el de un amigo, estoy en un apuro."],
        ["Por favor no le digas a nadie, me da mucha vergüenza con papi y mami.", "Es algo urgente pero te prometo que te lo devuelvo en cuanto pueda.", "Solo tú me puedes ayudar ahora mismo."],
        ["Necesito que me envíes el dinero por ATH Móvil o con tarjetas de regalo hoy mismo.", "Un señor del bufete te va a llamar para que le des los datos de tu tarjeta.", "Mándame {amount} a este número y te explico después."],
        ["Gracias, de verdad, eres la mejor, te quiero mucho.", "Te lo agradezco muchísimo, luego te llamo para contarte todo.", "No le comentes a nadie por ahora, te lo ruego."]]),
    ("romance_en", "en", [
        ["Hi sweetheart, it's me, {agent}, I finally got a signal to call you.", "Hello my dear, it's {agent}, I've been thinking about you all day.", "Hey love, it's {agent}, sorry it took me a few days to call."],
        ["I'm stuck at the port in another country and my card was blocked.", "My business trip has gone wrong and the customs office is holding my equipment.", "I had a small emergency with my health and the hospital needs a deposit."],
        ["You are the only person I trust and I'd never ask if it weren't important.", "I promise this is the last time I ask, and we'll be together soon.", "I feel so close to you and I don't know who else to turn to."],
        ["Could you send {amount} through a money transfer or gift cards and I'll pay you back when I land?", "If you can wire the money today I can finally come visit you.", "Please send it to my friend's account, I'll text you the details."],
        ["Thank you, my love, you've saved me.", "I can't wait to see you, you are wonderful.", "Please keep this between us until I get home."]]),
    ("invest_en", "en", [
        ["Good morning, this is {agent} from a private wealth management firm.", "Hello, my name is {agent}, I'm a senior investment advisor.", "Hi, {agent} calling, I help professionals in {city} grow their savings."],
        ["We have a limited program that returned over thirty percent last quarter.", "Our clients have been doubling their money with a new digital currency platform.", "There is an exclusive opportunity that is only open to a small group of residents."],
        ["It's completely safe and guaranteed, and you can withdraw whenever you want.", "Everything is regulated and the risk is practically zero.", "There is no obligation, I just want to share it with you."],
        ["The minimum to start is {amount}, which you can transfer today to secure your place.", "I'll send you a link to open your account and upload a photo of your ID.", "Wire the funds to our trading account and I'll activate your profile right away."],
        ["You'll see the first returns within a week, I promise.", "Thank you for trusting me with this, it's a great decision.", "Spots are filling quickly so don't wait too long."]]),
]

SAFE_CALLS = [
    ("bank_en", "en", [
        ["Hello, this is {agent} from the fraud monitoring team at {bank}, returning your call from earlier today.", "Good afternoon, my name is {agent} and I'm calling from {bank} card services about an alert on your account.", "Hi, this is {agent} with {bank}'s account protection department, following up on the alert you received."],
        ["We noticed a purchase of {amount} in {city} on your debit card ending in {last4} and wanted to check whether it was you.", "Our system flagged a payment of {amount} at an online store and we like to confirm unusual charges with our customers.", "There was a transfer of {amount} from your checking account this morning that we wanted to double check."],
        ["If you recognize it, there's nothing else to do and the card stays active.", "If you don't recognize it, I'll note the dispute and the card will be blocked automatically.", "Either way you won't be charged for anything you didn't authorize."],
        ["Please don't share any codes, PINs or passwords with me, we will never ask for those by phone.", "For your security I won't ask for your full card number or PIN, and if you'd like to verify me you can hang up and call the number on the back of your card.", "You can also check the activity in the {bank} mobile app, where this charge shows as pending."],
        ["Thanks for confirming, and have a good afternoon.", "I'll leave a note on your account, and you can call us any time at the number on your card.", "A replacement card will arrive by mail in five to seven business days, no action is needed from you."]]),
    ("bank_es", "es", [
        ["Buenas tardes, le habla {agent} del equipo de monitoreo de fraude de {bank}, devolviendo su llamada de hoy.", "Hola, mi nombre es {agent} y llamo de servicios de tarjetas de {bank} sobre una alerta en su cuenta.", "Saludos, soy {agent} de {bank}, le doy seguimiento a la alerta que usted recibió."],
        ["Vimos una compra de {amount} en {city} con su tarjeta terminada en {last4} y queríamos saber si fue usted.", "Nuestro sistema marcó un pago de {amount} y acostumbramos confirmar los cargos inusuales con el cliente.", "Hubo una transferencia de {amount} desde su cuenta esta mañana que queríamos revisar con usted."],
        ["Si la reconoce no tiene que hacer nada y la tarjeta sigue activa.", "Si no la reconoce, dejo anotada la disputa y la tarjeta se bloquea automáticamente.", "De cualquier forma no se le cobrará nada que usted no haya autorizado."],
        ["Por favor no me dé códigos, PIN ni contraseñas, nunca se los pedimos por teléfono.", "Por su seguridad no le voy a pedir el número completo de la tarjeta, y si desea verificarme puede colgar y llamar al número del dorso de su tarjeta.", "También puede revisar el movimiento en la aplicación móvil de {bank}, donde el cargo aparece pendiente."],
        ["Gracias por confirmar, que tenga buenas tardes.", "Dejo una nota en su cuenta y puede llamarnos cuando quiera al número de su tarjeta.", "Le llegará una tarjeta nueva por correo en cinco a siete días laborables, no tiene que hacer nada."]]),
    ("toll_es", "es", [
        ["Buenos días, le habla {agent} del servicio al cliente de AutoExpreso, devolviendo su llamada.", "Hola, soy {agent} de AutoExpreso, me pidieron que le llamara por su tablilla.", "Buenas tardes, mi nombre es {agent}, atiendo cuentas de AutoExpreso."],
        ["Veo que su cuenta tiene un balance pendiente de {amount} por {n} cruces del mes pasado.", "Su tablilla muestra una advertencia porque el método de pago venció.", "Tengo aquí su solicitud para actualizar la tarjeta que paga la cuenta."],
        ["No hay ningún cargo por mora si lo actualiza antes del cierre de este mes.", "Puede resolverlo con calma, no hay penalidad ni multa si lo hace esta semana.", "Es un trámite sencillo y puede hacerlo como más le convenga."],
        ["Lo puede hacer en la aplicación o en la página oficial de AutoExpreso, nosotros nunca le pedimos datos de la tarjeta por mensaje de texto.", "Si prefiere, visita una de nuestras oficinas con su tablilla y su licencia.", "También puede pagar en línea o llamar al número que aparece en su factura."],
        ["Gracias por llamar, que tenga un buen día.", "Quedo a la orden para cualquier duda, tome nota de este número de caso.", "Perfecto, ya dejé anotado todo en su cuenta."]]),
    ("pharmacy_en", "en", [
        ["Hello, this is {agent} from the pharmacy at the store in {city}.", "Hi, {agent} calling from your neighborhood pharmacy.", "Good morning, this is the automated refill line for your pharmacy, my name is {agent}."],
        ["Your prescription is ready for pick up and you can press 1 to confirm or press 2 to hear this message again.", "We're calling to confirm your refill and to verify the date of birth on file when you come in.", "Your insurance needs a quick update before we can fill it, so please bring your new card."],
        ["There is a copay of {amount} that you can pay at the counter.", "Please remember to bring a photo ID, it's required by law for this medication.", "There is no rush, we will hold it for seven days."],
        ["We will never ask for your card number or password over the phone.", "You can also confirm in our app or by calling the store directly.", "If you did not request this refill please call the store and let us know."],
        ["Thank you, and have a wonderful day.", "We're open until nine tonight.", "Thanks for choosing our pharmacy."]]),
    ("pharmacy_es", "es", [
        ["Buenos días, le habla {agent} de la farmacia en {city}.", "Hola, soy {agent} de su farmacia de la comunidad.", "Buenas tardes, esta es la línea de resurtidos de su farmacia, le habla {agent}."],
        ["Su receta está lista para recoger y puede oprimir 1 para confirmar o 2 para escuchar de nuevo este mensaje.", "Le llamamos para confirmar su resurtido y para verificar la fecha de nacimiento que tenemos cuando venga.", "Su plan médico necesita una actualización antes de despachar, por favor traiga su tarjeta nueva."],
        ["Hay un copago de {amount} que puede pagar en el mostrador.", "Recuerde traer una identificación con foto, es requisito para este medicamento.", "No hay prisa, se la guardamos por siete días."],
        ["Nunca le pedimos el número de su tarjeta ni contraseñas por teléfono.", "También puede confirmar en nuestra aplicación o llamando directamente a la tienda.", "Si usted no pidió este resurtido, por favor llame a la tienda y avísenos."],
        ["Gracias y que tenga un buen día.", "Estamos abiertos hasta las nueve de la noche.", "Gracias por escoger nuestra farmacia."]]),
    ("delivery_en", "en", [
        ["Hi, this is {agent} from the delivery service, I'm the driver on your route today.", "Hello, {agent} here from the courier company, I have a package for you.", "Good afternoon, this is {agent}, your delivery driver."],
        ["I'm about ten minutes from your address in {city} with a package that needs a signature.", "I tried to leave the package but the building requires someone to sign for it.", "Your package is on the truck and it should arrive {time}."],
        ["If nobody is home I can leave it with a neighbor or you can pick it up at our local office.", "You can track the shipment on the official app using the number in your confirmation email.", "There is no extra charge, the shipping was already paid by the sender."],
        ["Please don't send any payment to anyone claiming to be us, we never charge fees by phone.", "I just need a name for the signature, nothing else, no card or ID numbers.", "If you'd like, confirm your address so I can verify I'm on the right street."],
        ["Great, see you in a few minutes.", "Thanks and have a good one.", "Perfect, I'll note it in the system."]]),
    ("delivery_es", "es", [
        ["Hola, le habla {agent} del servicio de entregas, soy el conductor de su ruta hoy.", "Buenas tardes, soy {agent} de la compañía de mensajería, tengo un paquete para usted.", "Buenos días, soy {agent}, su repartidor."],
        ["Estoy a diez minutos de su dirección en {city} con un paquete que requiere firma.", "Intenté dejar el paquete pero el edificio requiere que alguien firme.", "Su paquete va en el camión y debe llegar {time}."],
        ["Si no hay nadie puedo dejarlo con un vecino o lo puede recoger en nuestra oficina local.", "Puede rastrear el envío en la aplicación oficial con el número de su correo de confirmación.", "No hay ningún cargo extra, el envío ya lo pagó quien lo mandó."],
        ["Por favor no envíe pagos a nadie que diga ser de la compañía, nunca cobramos cargos por teléfono.", "Solo necesito un nombre para la firma, nada más, ni tarjetas ni números de identificación.", "Si desea, confírmeme su dirección para verificar que estoy en la calle correcta."],
        ["Perfecto, nos vemos en unos minutos.", "Gracias y que tenga un buen día.", "Muy bien, lo anoto en el sistema."]]),
    ("utility_es", "es", [
        ["Buenos días, le llama {agent} de LUMA Energy con un aviso del servicio.", "Hola, soy {agent}, le llamamos para informarle de un trabajo programado en su área.", "Saludos, mi nombre es {agent}, de servicio al cliente de energía eléctrica."],
        ["Habrá una interrupción planificada del servicio {time} por mantenimiento en {city}.", "Su factura de este mes ya está disponible en línea por un monto de {amount}.", "Recibimos su solicitud de cambio de dirección y queríamos confirmarla."],
        ["El corte durará aproximadamente tres horas y no tiene que hacer nada.", "La fecha límite de pago es dentro de quince días, no hay cargos por mora todavía.", "Si ya pagó, puede ignorar este aviso."],
        ["Nunca le pediremos datos de su tarjeta por llamada, el pago se hace solo en la página oficial o en nuestras oficinas.", "Si tiene dudas, llámenos al número impreso en su factura.", "Este aviso es solo informativo."],
        ["Gracias por su comprensión, que tenga un buen día.", "Disculpe las molestias.", "Quedamos a la orden."]]),
    ("tax_en", "en", [
        ["Hello, this is {agent} from your tax preparation office, returning your message.", "Hi, {agent} calling from the accounting office in {city}.", "Good morning, this is {agent} from your tax preparer."],
        ["Your refund paperwork is ready and your federal refund is estimated at {amount}.", "We need to verify a few documents before we file your return.", "I'm calling to confirm your appointment {time}."],
        ["Please bring your photo ID and your social security card to the office so we can finish the file.", "We have the forms ready for your signature and nothing is due today.", "If you have any new W-2 forms you can drop them off or scan them to our portal."],
        ["We never ask for your social security number over the phone, we'll collect it in the office.", "You can pay our fee at the appointment, we don't take payments by gift card or wire.", "You can verify this call by phoning the office number on our website."],
        ["Great, see you then.", "Thank you, have a nice day.", "We'll keep your file ready."]]),
    ("appointment_es", "es", [
        ["Buenos días, le llamamos de la clínica dental en {city}.", "Hola, soy {agent} de la oficina de su médico de cabecera.", "Buenas tardes, le habla {agent} del centro de citas."],
        ["Queremos confirmar su cita {time}, por favor oprima 1 para confirmar o 2 para cancelar.", "Su cita está programada {time} y necesitamos verificar su plan médico.", "Notamos que hay un balance pendiente de {amount} de su última visita."],
        ["Por favor llegue quince minutos antes y traiga su tarjeta del plan.", "Si necesita cambiar la cita, puede hacerlo sin cargo hasta el día antes.", "El balance se puede pagar el día de la cita, no hay ninguna prisa."],
        ["Nunca le pedimos números de tarjeta por teléfono, el pago se hace solo en la oficina.", "Puede confirmar también por la aplicación de la clínica.", "Si no es usted la persona indicada, por favor déjenos saber."],
        ["Gracias, que tenga un excelente día.", "Lo esperamos.", "Muchas gracias por su tiempo."]]),
    ("inbound_bank_en", "en", [
        ["Thank you for calling {bank}, my name is {agent}, how can I help you today?", "Good afternoon, {bank} customer service, this is {agent}.", "Hello and thanks for calling {bank}, you're speaking with {agent}."],
        ["I can help with that, for your security I need to verify your identity first since you called us.", "Sure, I see the transaction of {amount} on your account from {city}.", "I understand you want to dispute a charge on the card ending in {last4}."],
        ["Could you confirm the last four digits of your social security number and the answer to your security question?", "Please confirm your date of birth and the phone number we have on file.", "I'll send a one-time code to the phone on file, and you can read it to me to continue."],
        ["Thank you, you're verified. I've opened the dispute and you'll see a temporary credit in two business days.", "I've blocked the old card and a new one will be mailed to your address.", "I've updated the account and I'll email you a summary of what we did."],
        ["Is there anything else I can help you with today?", "Thank you for being a {bank} customer.", "Have a great day and thanks for calling."]]),
    ("family_en", "en", [
        ["Hi mom, it's me, I'm at the store in {city}.", "Hey dad, I'm calling from the road, got a minute?", "Hi grandma, it's me, calling to say hello."],
        ["I'm picking up a birthday present and wanted to ask if you'd like a gift card for Maria or something else.", "I'm buying the gift card for the teacher and wanted to check the amount with you.", "I was thinking of sending a little money through ATH Móvil for the party, how does {amount} sound?"],
        ["It's no rush, I'll pay at the register and we'll settle up on Sunday.", "Don't worry about it, it's a small thing, just tell me what you prefer.", "I'll send you a photo of the options and you can pick."],
        ["Are you coming to the barbecue {time}? Everyone will be there.", "Do you need anything from the pharmacy while I'm out?", "Please say hi to everybody for me."],
        ["Love you, talk to you soon.", "Okay, see you then, bye.", "Thanks, call me back when you can."]]),
    ("insurance_en", "en", [
        ["Hello, this is {agent} from your insurance agency in {city}.", "Hi, {agent} calling from the customer service team of your auto insurance.", "Good afternoon, I'm {agent} from the renewals department."],
        ["Your policy renews next month and your premium would be {amount}.", "We wanted to let you know about an account update and confirm your mailing address.", "Our records show you qualify for a safe driver discount."],
        ["You don't need to do anything today, we just wanted to give you notice.", "If you'd like to review the coverage we can schedule a call {time}.", "You can also compare plans on our website at your own pace."],
        ["We will never ask you for your full card number or your password over the phone.", "If you want to make a payment, please use the official app or the number on your bill.", "You can verify who we are by calling the agency number printed on your policy."],
        ["Thank you for being a customer.", "Have a wonderful afternoon.", "Feel free to call back anytime."]]),
    ("school_es", "es", [
        ["Buenos días, le llama {agent} de la oficina de la escuela en {city}.", "Hola, soy {agent}, la secretaria de la escuela.", "Buenas tardes, llamo de la dirección escolar."],
        ["Queremos confirmar que recibió la notificación de la reunión de padres {time}.", "Necesitamos verificar los datos de contacto de emergencia de su hijo.", "Hay un balance de {amount} de la excursión de este mes."],
        ["Es solo un recordatorio, no tiene que enviar nada hoy.", "Puede actualizar los datos en la oficina o por la plataforma oficial de la escuela.", "El pago puede hacerlo con la maestra, en efectivo o con cheque."],
        ["La escuela nunca le pedirá números de tarjeta ni contraseñas por teléfono.", "Si tiene dudas puede llamar al número oficial de la escuela.", "Este es un aviso informativo."],
        ["Gracias por su tiempo, que tenga un buen día.", "Le esperamos en la reunión.", "Muchas gracias por su colaboración."]]),
]

# ===================================================================== SMS
FRAUD_SMS = [
    ("sms_toll_es", "es", [["AutoExpreso:", "AVISO AutoExpreso:", "Estimado cliente de AutoExpreso,"],
                           ["su cuenta tiene un balance pendiente de {amount}.", "tiene {n} cruces sin pagar en su tablilla.", "su tablilla será desactivada por falta de pago."],
                           ["Pague hoy para evitar la multa:", "Evite recargos y regularice su cuenta aquí:", "Actualice su método de pago en:"], ["{fake}", "{fake}", "Reply YES to continue.", ""]]),
    ("sms_toll_en", "en", [["AutoExpreso notice:", "Your AutoExpreso account:", "AutoExpreso billing:"],
                           ["you have an unpaid balance of {amount}.", "we found {n} toll crossings not charged to your tag.", "your tag will be suspended after a failed payment."],
                           ["Settle today to avoid penalties:", "Please update your payment method here:", "Avoid late fees by paying at:"], ["{fake}", "{fake}", "Reply YES to continue.", ""]]),
    ("sms_bank_es", "es", [["{bank}:", "Alerta de {bank}:", "Aviso de seguridad {bank}:"],
                           ["detectamos actividad inusual en su cuenta terminada en {last4}.", "su acceso a la banca en línea fue limitado temporalmente.", "se intentó un pago de {amount} con su tarjeta."],
                           ["Verifique su identidad para restablecer el acceso:", "Confirme que fue usted entrando aquí:", "Desbloquee su cuenta en el siguiente enlace:"], ["{fake}", "{fake}", "Reply YES to continue.", ""]]),
    ("sms_bank_en", "en", [["{bank} alert:", "{bank} security:", "Notice from {bank}:"],
                           ["we noticed unusual activity on your account ending in {last4}.", "your online banking access was temporarily limited.", "a payment of {amount} was attempted with your card."],
                           ["Verify your identity to restore access:", "Confirm it was you by signing in here:", "Unlock your account at the link below:"], ["{fake}", "{fake}", "Reply YES to continue.", ""]]),
    ("sms_ath_es", "es", [["ATH Móvil:", "Soporte ATH Móvil:", "Aviso ATH Móvil:"],
                          ["su cuenta quedará limitada por falta de validación.", "recibió un envío de {amount} pendiente de aceptar.", "hubo un intento de acceso desde otro dispositivo."],
                          ["Valide sus datos en:", "Acepte el envío o cancele aquí:", "Proteja su cuenta ahora:"], ["{fake}", "{fake}", "Reply YES to continue.", ""]]),
    ("sms_luma_es", "es", [["LUMA:", "LUMA Energy aviso:", "Factura LUMA:"],
                           ["su factura de {amount} está vencida.", "su servicio será suspendido hoy por falta de pago.", "tiene un ajuste pendiente en su cuenta de luz."],
                           ["Pague ahora para evitar el corte:", "Evite la suspensión pagando aquí:", "Actualice su pago en:"], ["{fake}", "{fake}", "Reply YES to continue.", ""]]),
    ("sms_usps_en", "en", [["USPS:", "Delivery notice:", "Parcel update:"],
                           ["your package could not be delivered because the address is incomplete.", "your parcel is on hold and a redelivery fee of {amount} is required.", "we attempted delivery in {city} but nobody was available."],
                           ["Confirm your address and pay the fee here:", "Reschedule delivery at:", "Update your delivery details:"], ["{fake}", "{fake}", "Reply YES to continue.", ""]]),
    ("sms_family_en", "en", [["Hi mom, it's me.", "Hello dad, this is my new number.", "Hey, it's your son, I broke my phone."],
                             ["I lost my phone and I'm texting from a friend's.", "I need help with a payment that is due today and my bank app is locked.", "I'm in a bit of a bind and I don't want to worry anyone."],
                             ["Can you send {amount} to this account for me?", "Could you transfer {amount} through ATH Móvil to this number?", "Please send {amount} in gift cards and I'll pay you back."],
                             ["I'll explain everything later, thank you!", "Please don't call, my phone is broken.", "You're the best."]]),
    ("sms_amazon_en", "en", [["Amazon:", "Order update:", "Your Amazon account:"],
                             ["your order of {amount} has been placed from a new device.", "there was a problem with your payment method and your order is on hold.", "we noticed a login from a new location."],
                             ["If this wasn't you, cancel here:", "Update your payment details here:", "Review the activity here:"], ["{fake}", "{fake}", "Reply YES to continue.", ""]]),
    ("sms_prize_es", "es", [["Felicidades:", "Aviso importante:", "Gran noticia:"],
                            ["fue seleccionado para un premio de {amount} en tarjetas de regalo.", "ganó un crédito por cliente frecuente de {amount}.", "tiene un reembolso de {amount} disponible."],
                            ["Reclámelo en:", "Complete el formulario aquí:", "Confirme sus datos para recibirlo:"], ["{fake}", "{fake}", "Reply YES to continue.", ""]]),
]
SAFE_SMS = [
    ("sms_otp_en", "en", [["{bank}:", "Your {bank} code:", "{bank} security code:"],
                          ["{code} is your verification code.", "your one-time code is {code}.", "use {code} to confirm your sign in."],
                          ["Never share it with anyone, we will never ask for it.", "It expires in 10 minutes.", "If you didn't request it, ignore this message."], ["{real}", "{real}", "", ""]]),
    ("sms_otp_es", "es", [["{bank}:", "Su código de {bank}:", "Código de seguridad {bank}:"],
                          ["{code} es su código de verificación.", "su código de un solo uso es {code}.", "use {code} para confirmar su acceso."],
                          ["No lo comparta con nadie, nunca se lo pediremos.", "Expira en 10 minutos.", "Si no lo solicitó, ignore este mensaje."], ["{real}", "{real}", "", ""]]),
    ("sms_appt_en", "en", [["Reminder:", "Appointment reminder:", "Hi {name},"],
                           ["you have an appointment {time} at our office in {city}.", "your visit is scheduled {time}, please arrive 10 minutes early.", "we're confirming your appointment {time}."],
                           ["Reply C to confirm or R to reschedule.", "Manage it at {real}.", "Need to change it? Call the office."], ["{real}", "{real}", "", ""]]),
    ("sms_appt_es", "es", [["Recordatorio:", "Recordatorio de cita:", "Hola {name},"],
                           ["tiene una cita {time} en nuestra oficina en {city}.", "su visita está programada {time}, llegue 10 minutos antes.", "le confirmamos su cita {time}."],
                           ["Responda C para confirmar o R para reprogramar.", "Puede cambiarla en {real}.", "¿Necesita cambiarla? Llame a la oficina."], ["{real}", "{real}", "", ""]]),
    ("sms_delivery_en", "en", [["UPS:", "Delivery update:", "Your package:"],
                               ["your package will arrive {time}.", "your delivery is out for delivery today in {city}.", "a package was delivered to your front door."],
                               ["Track it at {real}.", "Follow your delivery at {real}.", "No action is needed."], ["{real}", "{real}", "", ""]]),
    ("sms_delivery_es", "es", [["Aviso de entrega:", "Su paquete:", "Actualización de envío:"],
                               ["su paquete llegará {time}.", "su entrega va en camino hoy en {city}.", "un paquete fue entregado en su puerta."],
                               ["Rastréelo en {real}.", "Siga su envío en {real}.", "No tiene que hacer nada."], ["{real}", "{real}", "", ""]]),
    ("sms_luma_es", "es", [["LUMA:", "LUMA Energy:", "Aviso de servicio:"],
                           ["su factura de {amount} ya está disponible.", "habrá mantenimiento programado {time} en {city}.", "su pago fue recibido, gracias."],
                           ["Revísela en la página oficial.", "El servicio volverá en unas horas.", "Guarde este mensaje como comprobante."], ["{real}", "{real}", "", ""]]),
    ("sms_ath_es", "es", [["ATH Móvil:", "ATH Móvil recibo:", "Notificación:"],
                          ["recibió {amount} de {name}.", "su envío de {amount} a {name} fue completado.", "{name} le envió {amount}."],
                          ["Revise su balance en la aplicación.", "Gracias por usar ATH Móvil.", "No responda a este mensaje."], ["{real}", "{real}", "", ""]]),
    ("sms_pharmacy_en", "en", [["Pharmacy:", "Refill update:", "Hi {name},"],
                               ["your prescription is ready for pickup.", "your refill is ready, please bring your ID.", "we need to verify your date of birth when you pick up your order."],
                               ["We're open until 9 PM.", "Reply STOP to opt out.", "Details at {real}."], ["{real}", "{real}", "", ""]]),
    ("sms_work_en", "en", [["Urgent:", "Reminder:", "Heads up:"],
                           ["the team meeting moved to {time}.", "please confirm your shift for {time}.", "the account review for the client is due {time}."],
                           ["Reply to confirm.", "Let me know if you can't make it.", "Check the shared calendar."], ["{real}", "{real}", "", ""]]),
]

# ===================================================================== EMAIL
FRAUD_EMAIL = [
    ("mail_invoice_en", "en", [["Subject: Invoice {code} attached", "Subject: Payment confirmation for your order", "Subject: Your invoice is overdue"],
                               ["Hello {name},", "Dear customer,", "Hi {name},"],
                               ["We could not process the payment of {amount} for your recent order, so your account has been placed on hold.", "Please find attached the invoice for {amount}, which is now past due.", "A payment of {amount} to your account was flagged and needs your review."],
                               ["To avoid service interruption, review the invoice and update your details here: {fake}", "Log in to confirm the payment and release the order: {fake}", "Open the secure portal to settle the balance: {fake}"],
                               ["Thank you for your prompt attention.", "Best regards, Billing Team", "If you have questions, reply to this email."]]),
    ("mail_password_en", "en", [["Subject: Your password expires today", "Subject: Action required: confirm your account", "Subject: Security alert for your account"],
                                ["Dear user,", "Hello {name},", "Hi,"],
                                ["Your email password will expire in 24 hours and your mailbox may be suspended.", "We detected a sign-in from a new device in {city}.", "Your account needs to be verified to keep your storage and messages."],
                                ["Keep your current password by confirming here: {fake}", "If this wasn't you, secure your account now: {fake}", "Verify now to avoid losing access: {fake}"],
                                ["IT Support Team", "Security Department", "Thank you for helping us keep your account safe."]]),
    ("mail_bank_es", "es", [["Asunto: Alerta de seguridad en su cuenta {bank}", "Asunto: Verifique su identidad", "Asunto: Su cuenta ha sido limitada"],
                            ["Estimado cliente:", "Hola {name}:", "Apreciado cliente de {bank}:"],
                            ["Detectamos un intento de acceso a su banca en línea desde un dispositivo no reconocido.", "Por motivos de seguridad hemos limitado temporalmente su cuenta terminada en {last4}.", "Se registró un cargo de {amount} que necesita su confirmación."],
                            ["Para restablecer el acceso verifique sus datos en: {fake}", "Ingrese aquí para confirmar que fue usted: {fake}", "Complete la validación en el siguiente enlace: {fake}"],
                            ["Atentamente, Departamento de Seguridad", "Gracias por su atención.", "Este mensaje es automático, no responda."]]),
    ("mail_hacienda_es", "es", [["Asunto: Reembolso de contribuciones disponible", "Asunto: Notificación de Hacienda", "Asunto: Su planilla tiene un ajuste pendiente"],
                                ["Estimado contribuyente:", "Hola {name}:", "Saludos:"],
                                ["Según nuestros registros usted tiene un reembolso de {amount} pendiente de procesar.", "Su planilla presenta un ajuste que debe atenderse para evitar recargos.", "Encontramos un dato incompleto en su declaración."],
                                ["Complete el formulario para recibir el depósito: {fake}", "Revise y confirme su información aquí: {fake}", "Acceda al portal para resolverlo: {fake}"],
                                ["Departamento de Reembolsos", "Gracias por su cooperación.", "Este es un mensaje oficial."]]),
    ("mail_package_en", "en", [["Subject: Delivery attempt failed", "Subject: Your parcel is waiting", "Subject: Action needed to deliver your package"],
                               ["Hello {name},", "Dear customer,", "Hi,"],
                               ["We tried to deliver your parcel in {city} but the address was incomplete.", "Your package is held at our facility until a small fee of {amount} is paid.", "Your shipment will be returned if we don't hear from you."],
                               ["Confirm your address and pay the fee at: {fake}", "Schedule redelivery here: {fake}", "Update your delivery information: {fake}"],
                               ["Customer Service", "Thank you for your patience.", "Delivery Support Team"]]),
    ("mail_share_en", "en", [["Subject: {name} shared a document with you", "Subject: Shared file: Q3 report", "Subject: You have a new secure message"],
                             ["Hi,", "Hello {name},", "Dear colleague,"],
                             ["A document was shared with you and requires your review today.", "You have a secure message waiting from your manager.", "A file has been shared with you from the finance team."],
                             ["Sign in with your work account to open it: {fake}", "Open the document here: {fake}", "View the message in your secure inbox: {fake}"],
                             ["Thanks.", "Sent from my phone.", "Please let me know once you've reviewed it."]]),
]
SAFE_EMAIL = [
    ("mail_order_en", "en", [["Subject: Your order has shipped", "Subject: Order confirmation", "Subject: Thanks for your purchase"],
                             ["Hi {name},", "Hello {name},", "Dear {name},"],
                             ["Thank you for your order, your total was {amount} and it ships {time}.", "Your payment of {amount} was received and your order is being prepared.", "Your items are on their way to your address in {city}."],
                             ["You can track your order in your account at {real}.", "View the details at {real}, no action is needed.", "If anything looks wrong, contact our support team through the official site."],
                             ["Thanks for shopping with us.", "Customer Service Team", "We hope you enjoy your purchase."]]),
    ("mail_statement_en", "en", [["Subject: Your {bank} statement is ready", "Subject: Account alert from {bank}", "Subject: Monthly statement available"],
                                 ["Hello {name},", "Dear customer,", "Hi {name},"],
                                 ["Your latest statement is now available in online banking.", "We noticed a payment of {amount}, and this is a notification that it posted to your account.", "This message confirms the update you made to your contact information."],
                                 ["Sign in through the official {bank} website or app to view it. We will never ask for your password or full card number by email.", "If you did not make this change, call the number on the back of your card.", "Please do not reply to this email; it is sent from an unmonitored address."],
                                 ["Thank you for banking with us.", "{bank} Customer Care", "Sincerely, the {bank} team"]]),
    ("mail_news_en", "en", [["Subject: Last chance: 40% off this weekend only!", "Subject: URGENT: our biggest sale ends tonight", "Subject: Don't miss out on free shipping"],
                            ["Hi {name},", "Hello,", "Dear shopper,"],
                            ["Our weekend sale ends tonight, and you can save up to 40 percent on selected items.", "Get free shipping on orders over {amount} for a limited time.", "Your loyalty points of {n}00 expire at the end of the month, so use them on your next order."],
                            ["Shop now at {real}.", "Browse the offers at {real} and use code SAVE{n}.", "See the full details in our store."],
                            ["Unsubscribe any time using the link below.", "Thanks for being a customer.", "Happy shopping!"]]),
    ("mail_work_en", "en", [["Subject: Urgent: please review before {time}", "Subject: Action required: budget approval", "Subject: Fwd: contract signature needed"],
                            ["Hi team,", "Hello {name},", "Hi all,"],
                            ["Please confirm your availability for the client review {time}.", "I need your approval on the attached budget of {amount} before the deadline.", "Could you verify the numbers in the shared spreadsheet and send me comments?"],
                            ["The files are in the shared drive we use every week.", "Let me know if you have access issues and I'll ask IT.", "You can reach me at my desk or by phone if it's urgent."],
                            ["Thanks.", "Best, {agent}", "Appreciate it."]]),
    ("mail_appt_es", "es", [["Asunto: Confirmación de su cita", "Asunto: Recordatorio de cita médica", "Asunto: Su cita ha sido programada"],
                            ["Hola {name}:", "Estimado paciente:", "Buenos días {name}:"],
                            ["Su cita está programada {time} en nuestra clínica en {city}.", "Le recordamos su visita {time}, por favor llegue con 15 minutos de anticipación.", "Hemos recibido su solicitud y su cita fue confirmada."],
                            ["Puede reprogramar en nuestra página oficial {real} o llamando a la oficina.", "Traiga su tarjeta del plan médico y una identificación. No le pediremos datos por correo.", "Si no puede asistir, avísenos con 24 horas de anticipación."],
                            ["Gracias por su confianza.", "Atentamente, la oficina", "Que tenga un buen día."]]),
    ("mail_utility_es", "es", [["Asunto: Su factura de energía está disponible", "Asunto: Aviso de mantenimiento programado", "Asunto: Confirmación de pago recibido"],
                               ["Estimado cliente:", "Hola {name}:", "Saludos:"],
                               ["Su factura de este mes por {amount} ya está disponible en su cuenta en línea.", "Habrá mantenimiento programado {time} en {city} y el servicio puede interrumpirse brevemente.", "Recibimos su pago de {amount}, gracias."],
                               ["Puede verla en el portal oficial {real}. No pedimos datos de tarjeta por correo.", "Si ya pagó, ignore este aviso.", "Para dudas, llame al número impreso en su factura."],
                               ["Atentamente, Servicio al Cliente", "Gracias por su comprensión.", "Este es un mensaje automático."]]),
]


# Balance the shortcuts: some fraud emails carry no link (they ask for a reply), and safe emails often link to the real site.
NO_LINK_FRAUD = {"en": "Reply to this email with your full name, date of birth and account number and we will fix it for you.",
                 "es": "Responda a este correo con su nombre, fecha de nacimiento y número de cuenta y lo resolvemos por usted."}
MORE_DETAILS_SAFE = {"en": "More details are available at {real}.", "es": "Más detalles en {real}."}
for _sid, _lang, _steps in FRAUD_EMAIL:
    _steps[3].append(NO_LINK_FRAUD[_lang])
for _sid, _lang, _steps in SAFE_EMAIL:
    _steps[3].append(MORE_DETAILS_SAFE[_lang])


def _tidy(text: str) -> str:
    """Remove artefacts of the assembly: a dangling 'here:' and a link written twice."""
    text = re.sub(r":\s*$", ".", text.strip())
    seen = set()

    def once(m):
        u = m.group(0).rstrip(".")
        if u in seen:
            return ""
        seen.add(u)
        return m.group(0)
    text = re.sub(r"https?://\S+", once, text)
    text = re.sub(r"\s+\.", ".", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _build(skeletons, kind: str, label: int, n_per: int, rng: random.Random) -> list[dict]:
    rows = []
    for sid, lang, steps in skeletons:
        seen = set()
        for k in range(n_per * 3):                      # draw extra, keep distinct
            slots = _slots(rng, lang)
            slots["fake"] = _scenario_link(sid, SCENARIO_LINKS, FAKE_LINKS, rng)
            slots["real"] = _scenario_link(sid, SCENARIO_REAL, REAL_LINKS, rng)
            parts = [rng.choice(step).format(**slots) for step in steps]
            text = _tidy(("\n" if kind == "email" else " ").join(p for p in parts if p))
            if text in seen:
                continue
            seen.add(text)
            rows.append({"kind": kind, "text": text, "label": label, "lang": lang, "group": f"hard:{kind}:{label}:{sid}",
                         "source": f"hard_{kind}"})   # class-neutral on purpose
            if len(seen) >= n_per:
                break
    return rows


def generate_hard_set(seed: int = C.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(calls, messages) in the loaders' schema plus a ``group`` column (one group per skeleton)."""
    rng = random.Random(seed)
    n = C.HARD_PER_SKELETON
    calls = _build(FRAUD_CALLS, "call", 1, n["call"], rng) + _build(SAFE_CALLS, "call", 0, n["call"], rng)
    sms = _build(FRAUD_SMS, "sms", 1, n["sms"], rng) + _build(SAFE_SMS, "sms", 0, n["sms"], rng)
    email = _build(FRAUD_EMAIL, "email", 1, n["email"], rng) + _build(SAFE_EMAIL, "email", 0, n["email"], rng)
    c = pd.DataFrame(calls).rename(columns={"kind": "_k"}).drop(columns="_k")
    c.insert(0, "call_id", [f"hardcall_{i}" for i in range(len(c))])
    m = pd.DataFrame(sms + email)
    m.insert(0, "item_id", [f"hardmsg_{i}" for i in range(len(m))])
    return c[["call_id", "text", "label", "source", "lang", "group"]], m[["item_id", "kind", "text", "label", "source", "lang", "group"]]
