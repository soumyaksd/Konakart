package com.konakart.app;

import com.konakart.appif.*;

/**
 *  The KonaKart Custom Engine - AddCreditCardToCustomer - Generated by CreateKKCustomEng
 */
@SuppressWarnings("all")
public class AddCreditCardToCustomer
{
    KKEng kkEng = null;

    /**
     * Constructor
     */
     public AddCreditCardToCustomer(KKEng _kkEng)
     {
         kkEng = _kkEng;
     }

     public int addCreditCardToCustomer(String sessionId, CreditCardIf card, CreditCardOptionsIf options) throws Exception
     {
         return kkEng.addCreditCardToCustomer(sessionId, card, options);
     }
}