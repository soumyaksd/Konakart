package com.konakartadmin.app;

import com.konakartadmin.appif.*;
import com.konakartadmin.bl.KKAdmin;

/**
 *  The KonaKart Custom Engine - GetLanguageById - Generated by CreateKKAdminCustomEng
 */
@SuppressWarnings("all")
public class GetLanguageById
{
    KKAdmin kkAdminEng = null;

    /**
     * Constructor
     */
     public GetLanguageById(KKAdmin _kkAdminEng)
     {
         kkAdminEng = _kkAdminEng;
     }

     public AdminLanguage getLanguageById(int id) throws KKAdminException
     {
         return kkAdminEng.getLanguageById(id);
     }
}
