package com.konakart.app;

import com.konakart.appif.*;

/**
 *  The KonaKart Custom Engine - GetCreditCardsPerCustomer - Generated by CreateKKCustomEng
 */
@SuppressWarnings("all")
public class GetCreditCardsPerCustomer
{
    KKEng kkEng = null;

    /**
     * Constructor
     */
     public GetCreditCardsPerCustomer(KKEng _kkEng)
     {
         kkEng = _kkEng;
     }

     public CreditCardIf[] getCreditCardsPerCustomer(String sessionId, CreditCardOptionsIf options) throws Exception
     {
         return kkEng.getCreditCardsPerCustomer(sessionId, options);
     }
}
