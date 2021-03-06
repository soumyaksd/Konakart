package com.konakart.app;

import com.konakart.appif.*;

/**
 *  The KonaKart Custom Engine - GetAddressesPerStore - Generated by CreateKKCustomEng
 */
@SuppressWarnings("all")
public class GetAddressesPerStore
{
    KKEng kkEng = null;

    /**
     * Constructor
     */
     public GetAddressesPerStore(KKEng _kkEng)
     {
         kkEng = _kkEng;
     }

     public AddressIf[] getAddressesPerStore(String addressStoreId) throws KKException
     {
         return kkEng.getAddressesPerStore(addressStoreId);
     }
}
