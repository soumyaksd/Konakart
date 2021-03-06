package com.konakart.app;

import com.konakart.appif.*;

/**
 *  The KonaKart Custom Engine - GetAlsoPurchasedWithOptions - Generated by CreateKKCustomEng
 */
@SuppressWarnings("all")
public class GetAlsoPurchasedWithOptions
{
    KKEng kkEng = null;

    /**
     * Constructor
     */
     public GetAlsoPurchasedWithOptions(KKEng _kkEng)
     {
         kkEng = _kkEng;
     }

     public ProductIf[] getAlsoPurchasedWithOptions(String sessionId, DataDescriptorIf dataDesc, int productId, int languageId, FetchProductOptionsIf options) throws KKException
     {
         return kkEng.getAlsoPurchasedWithOptions(sessionId, dataDesc, productId, languageId, options);
     }
}
