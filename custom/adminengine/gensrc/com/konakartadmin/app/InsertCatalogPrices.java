package com.konakartadmin.app;

import com.konakartadmin.appif.*;
import com.konakartadmin.bl.KKAdmin;

/**
 *  The KonaKart Custom Engine - InsertCatalogPrices - Generated by CreateKKAdminCustomEng
 */
@SuppressWarnings("all")
public class InsertCatalogPrices
{
    KKAdmin kkAdminEng = null;

    /**
     * Constructor
     */
     public InsertCatalogPrices(KKAdmin _kkAdminEng)
     {
         kkAdminEng = _kkAdminEng;
     }

     public void insertCatalogPrices(String sessionId, AdminProductPrice[] prices, AdminInsertCatalogPriceOptions options) throws KKAdminException
     {
         kkAdminEng.insertCatalogPrices(sessionId, prices, options);
     }
}