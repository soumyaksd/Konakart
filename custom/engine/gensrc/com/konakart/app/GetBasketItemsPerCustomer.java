package com.konakart.app;

import com.konakart.appif.*;

/**
 *  The KonaKart Custom Engine - GetBasketItemsPerCustomer - Generated by CreateKKCustomEng
 */
@SuppressWarnings("all")
public class GetBasketItemsPerCustomer
{
    KKEng kkEng = null;

    /**
     * Constructor
     */
     public GetBasketItemsPerCustomer(KKEng _kkEng)
     {
         kkEng = _kkEng;
     }

     public BasketIf[] getBasketItemsPerCustomer(String sessionId, int customerId, int languageId) throws KKException
     {
         return kkEng.getBasketItemsPerCustomer(sessionId, customerId, languageId);
     }
}
