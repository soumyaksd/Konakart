package com.konakart.app;

import com.konakart.appif.*;

/**
 *  The KonaKart Custom Engine - EditCustomerWithOptions - Generated by CreateKKCustomEng
 */
@SuppressWarnings("all")
public class EditCustomerWithOptions
{
    KKEng kkEng = null;

    /**
     * Constructor
     */
     public EditCustomerWithOptions(KKEng _kkEng)
     {
         kkEng = _kkEng;
     }

     public void editCustomerWithOptions(String sessionId, CustomerIf cust, EditCustomerOptionsIf options) throws KKException
     {
         kkEng.editCustomerWithOptions(sessionId, cust, options);
     }
}
